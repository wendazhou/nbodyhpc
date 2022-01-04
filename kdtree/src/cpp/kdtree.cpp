#include "kdtree.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <vector>

#include "span.hpp"
#include "utils.hpp"

namespace wenda {

namespace kdtree {

template <typename T, size_t R>
T l2_distance(std::array<T, R> const &left, std::array<T, R> const &right) {
    T result = 0;

    for (size_t i = 0; i < R; ++i) {
        result += (left[i] - right[i]) * (left[i] - right[i]);
    }

    return result;
}

namespace {

struct MutexLockSynchronization {
    typedef std::mutex mutex_t;
    typedef std::scoped_lock<std::mutex> lock_t;
};

struct NullLock {
    template <typename... T> NullLock(T...) {}
};

struct NullSynchonization {
    typedef std::nullptr_t mutex_t;
    typedef NullLock lock_t;
};

template <typename Synchronization = MutexLockSynchronization> struct KDTreeBuilder {
    std::vector<KDTree::KDTreeNode> &nodes_;
    tcb::span<std::array<float, 3>> positions_;
    size_t leaf_size_;
    typename Synchronization::mutex_t mutex_;

    KDTreeBuilder(
        std::vector<KDTree::KDTreeNode> &nodes, tcb::span<std::array<float, 3>> positions,
        size_t leaf_size)
        : nodes_(nodes), positions_(positions), leaf_size_(leaf_size) {}

    void build(int thread_levels) {
        build_node(0, 0, static_cast<uint32_t>(positions_.size()), thread_levels);
    }

    uint32_t build_node(int dimension, uint32_t left, uint32_t count, int thread_levels) {
        typedef typename Synchronization::lock_t lock_t;

        auto positions = positions_.subspan(left, count);

        if (positions.size() < leaf_size_) {
            lock_t lock(mutex_);
            nodes_.push_back({-1, 0.0f, left, left + count});
            return static_cast<uint32_t>(nodes_.size() - 1);
        }

        auto median_it = positions.begin() + positions.size() / 2;
        std::nth_element(
            positions.begin(),
            median_it,
            positions.end(),
            [dimension](std::array<float, 3> const &a, std::array<float, 3> const &b) {
                return a[dimension] < b[dimension];
            });

        float split = (*median_it)[dimension];

        uint32_t current_idx;

        {
            lock_t lock(mutex_);
            current_idx = static_cast<uint32_t>(nodes_.size());
            nodes_.push_back({dimension, split, 0u, 0u});
        }

        uint32_t left_idx, right_idx;

        if (thread_levels > 0) {
            std::tie(left_idx, right_idx) =
                build_left_right_threaded(dimension, left, count, thread_levels - 1);
        } else {
            std::tie(left_idx, right_idx) = build_left_right_nonthreaded(dimension, left, count);
        }

        {
            lock_t lock(mutex_);
            nodes_[current_idx].left_ = left_idx;
            nodes_[current_idx].right_ = right_idx;
        }

        return current_idx;
    }

    std::pair<uint32_t, uint32_t>
    build_left_right_nonthreaded(int dimension, uint32_t left, uint32_t count) {
        std::pair<uint32_t, uint32_t> result;

        result.first = build_node((dimension + 1) % 3, left, count / 2, 0);
        result.second = build_node((dimension + 1) % 3, left + count / 2, count - count / 2, 0);

        return result;
    }

    std::pair<uint32_t, uint32_t>
    build_left_right_threaded(int dimension, uint32_t left, uint32_t count, int thread_levels) {
        std::pair<uint32_t, uint32_t> result;

        std::future<uint32_t> left_future = std::async(std::launch::async, [&]() {
            return build_node((dimension + 1) % 3, left, count / 2, thread_levels - 1);
        });

        result.second =
            build_node((dimension + 1) % 3, left + count / 2, count - count / 2, thread_levels - 1);

        result.first = left_future.get();
        return result;
    }
};

struct KDTreeQuery {
    tcb::span<const std::array<float, 3>> positions_;
    tcb::span<const KDTree::KDTreeNode> nodes_;
    std::array<float, 3> const &query_;
    std::priority_queue<float> distances_;
    size_t num_nodes_visited = 0;
    size_t num_nodes_pruned = 0;
    size_t num_points_visited = 0;

    KDTreeQuery(KDTree const &tree, std::array<float, 3> const &query, size_t k)
        : positions_(tree.positions()), nodes_(tree.nodes()), query_(query),
          distances_(std::less<float>{}, std::vector<float>(k, std::numeric_limits<float>::max())) {
    }

    bool push_point(std::array<float, 3> const &p) {
        auto dist = l2_distance(p, query_);

        if (dist < distances_.top()) {
            distances_.pop();
            distances_.push(dist);
            return true;
        }

        return false;
    }

    void compute(KDTree::KDTreeNode const *node) {
        num_nodes_visited += 1;

        if (node->dimension_ == -1) {
            tcb::span<const std::array<float, 3>> node_positions(
                positions_.data() + node->left_, positions_.data() + node->right_);

            for (auto const &p : node_positions) {
                push_point(p);
            }

            num_points_visited += node_positions.size();

            return;
        }

        auto closer = nodes_.data() + node->left_;
        auto further = nodes_.data() + node->right_;

        if (query_[node->dimension_] > node->split_) {
            std::swap(closer, further);
        }

        compute(closer);

        auto delta_dimension = query_[node->dimension_] - node->split_;

        if (distances_.top() < delta_dimension * delta_dimension) {
            num_nodes_pruned += 1;
            return;
        }

        compute(further);
    }
};

} // namespace

KDTree::KDTree(tcb::span<const std::array<float, 3>> positions, KDTreeConfiguration const &config)
    : positions_(positions.begin(), positions.end()) {

    if (positions.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("More than uint32_t points are not supported.");
    }

    const size_t num_points_per_thread = 5000000;
    int max_threads =
        config.max_threads == -1 ? std::thread::hardware_concurrency() : config.max_threads;

    if (max_threads > 1 && positions.size() >= 2 * num_points_per_thread) {
        double num_threads = static_cast<double>(positions.size()) / num_points_per_thread;
        num_threads = std::max(num_threads, static_cast<double>(max_threads));

        int log2_threads = static_cast<int>(std::log2(num_threads));

        typedef MutexLockSynchronization Synchronization;
        KDTreeBuilder<Synchronization> builder(nodes_, positions_, config.leaf_size);
        builder.build(log2_threads);
    } else {
        KDTreeBuilder<NullSynchonization> builder(nodes_, positions_, config.leaf_size);
        builder.build(0);
    }
}

std::vector<float> KDTree::find_closest(
    std::array<float, 3> const &position, size_t k, KDTreeQueryStatistics *statistics) const {

    KDTreeQuery query(*this, position, k);
    query.compute(&nodes_[0]);

    if (statistics) {
        statistics->nodes_visited = query.num_nodes_visited;
        statistics->nodes_pruned = query.num_nodes_pruned;
        statistics->points_visited = query.num_points_visited;
    }

    std::vector result(std::move(get_container_from_adapter(query.distances_)));
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace kdtree

} // namespace wenda
