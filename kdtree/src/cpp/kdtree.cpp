#include "kdtree.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <vector>

#include "utils.hpp"
#include <span.hpp>

#include <iostream>

namespace wenda {

namespace kdtree {

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

std::vector<PositionAndIndex>
make_position_and_indices(tcb::span<const std::array<float, 3>> const &positions) {
    std::vector<PositionAndIndex> result(positions.size());

    for (size_t i = 0; i < positions.size(); ++i) {
        result[i].position = positions[i];
        result[i].index = i;
    }

    return result;
}

template <typename Synchronization = MutexLockSynchronization> struct KDTreeBuilder {
    std::vector<KDTree::KDTreeNode> &nodes_;
    tcb::span<PositionAndIndex> positions_;
    size_t leaf_size_;
    typename Synchronization::mutex_t mutex_;

    KDTreeBuilder(
        std::vector<KDTree::KDTreeNode> &nodes, tcb::span<PositionAndIndex> positions,
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
            [dimension](PositionAndIndex const &a, PositionAndIndex const &b) {
                return a.position[dimension] < b.position[dimension];
            });

        float split = median_it->position[dimension];

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

struct PairLessFirst {
    bool operator()(
        std::pair<float, uint32_t> const &left, std::pair<float, uint32_t> const &right) const {
        return left.first < right.first;
    }
};

//! Utility structure used to store state of KD-tree search.
template <typename Distance = L2Distance> struct KDTreeQuery {
    typedef std::pair<float, uint32_t> result_t;

    Distance const &distance_;
    tcb::span<const PositionAndIndex> positions_;
    tcb::span<const KDTree::KDTreeNode> nodes_;
    std::array<float, 3> const &query_;
    std::priority_queue<result_t, std::vector<result_t>, PairLessFirst> distances_;
    size_t num_nodes_visited = 0;
    size_t num_nodes_pruned = 0;
    size_t num_points_visited = 0;

    KDTreeQuery(
        KDTree const &tree, Distance const &distance, std::array<float, 3> const &query, size_t k)
        : distance_(distance), positions_(tree.positions()), nodes_(tree.nodes()), query_(query),
          distances_(
              PairLessFirst{}, std::vector<result_t>(k, {std::numeric_limits<float>::max(), -1})) {}

    void process_leaf(KDTree::KDTreeNode const &node) {
        uint32_t const num_points = node.right_ - node.left_;
        auto node_positions = positions_.subspan(node.left_, num_points);

        float current_distance = distances_.top().first;

        for (uint32_t i = 0; i < num_points; ++i) {
            auto dist = distance_(node_positions[i].position, query_);

            if (dist >= current_distance) {
                continue;
            }

            distances_.pop();
            distances_.push({dist, node_positions[i].index});
            current_distance = distances_.top().first;
        }

        num_points_visited += node_positions.size();
    }

    void compute(KDTree::KDTreeNode const *node, std::array<float, 6> const &bounds) {
        num_nodes_visited += 1;

        if (node->dimension_ == -1) {
            process_leaf(*node);
            return;
        }

        auto closer = nodes_.data() + node->left_;
        auto further = nodes_.data() + node->right_;
        size_t close_boundary_dim = 2 * node->dimension_ + 1;
        size_t far_boundary_dim = 2 * node->dimension_;

        if (query_[node->dimension_] > node->split_) {
            std::swap(closer, further);
            std::swap(close_boundary_dim, far_boundary_dim);
        }

        {
            std::array<float, 6> close_bounds = bounds;
            close_bounds[close_boundary_dim] = node->split_;
            auto distance = distance_.box_distance(query_, close_bounds);

            if (distance < distances_.top().first) {
                compute(closer, close_bounds);
            } else {
                num_nodes_pruned += 1;
            }
        }

        std::array<float, 6> far_bounds = bounds;
        far_bounds[far_boundary_dim] = node->split_;

        auto distance = distance_.box_distance(query_, far_bounds);
        auto delta_dimension = query_[node->dimension_] - node->split_;

        if (distances_.top().first < distance) {
            num_nodes_pruned += 1;
            return;
        }

        compute(further, far_bounds);
    }
};

} // namespace

KDTree::KDTree(tcb::span<const std::array<float, 3>> positions, KDTreeConfiguration const &config)
    : KDTree(make_position_and_indices(positions), config) {}

KDTree::KDTree(
    std::vector<PositionAndIndex> &&positions_and_indices, KDTreeConfiguration const &config)
    : positions_(std::move(positions_and_indices)) {

    if (positions_.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("More than uint32_t points are not supported.");
    }

    const size_t num_points_per_thread = 5000000;
    int max_threads =
        config.max_threads == -1 ? std::thread::hardware_concurrency() : config.max_threads;

    if (max_threads > 1 && positions_.size() >= 2 * num_points_per_thread) {
        double num_threads = static_cast<double>(positions_.size()) / num_points_per_thread;
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

template <typename Distance>
std::vector<std::pair<float, uint32_t>> KDTree::find_closest(
    std::array<float, 3> const &position, size_t k, Distance const &distance,
    KDTreeQueryStatistics *statistics) const {

    KDTreeQuery<Distance> query(*this, distance, position, k);
    query.compute(&nodes_[0], distance.initial_box(position));

    if (statistics) {
        statistics->nodes_visited = query.num_nodes_visited;
        statistics->nodes_pruned = query.num_nodes_pruned;
        statistics->points_visited = query.num_points_visited;
    }

    std::vector result(std::move(get_container_from_adapter(query.distances_)));
    std::sort(result.begin(), result.end(), PairLessFirst{});

    // convert back to distances from distance-squared
    for (auto &p : result) {
        p.first = distance.postprocess(p.first);
    }

    return result;
}

template std::vector<std::pair<float, uint32_t>> KDTree::find_closest<L2Distance>(
    std::array<float, 3> const &position, size_t k, L2Distance const &,
    KDTreeQueryStatistics *statistics) const;

template std::vector<std::pair<float, uint32_t>> KDTree::find_closest<L2PeriodicDistance<float>>(
    std::array<float, 3> const &position, size_t k, L2PeriodicDistance<float> const &,
    KDTreeQueryStatistics *statistics) const;

} // namespace kdtree

} // namespace wenda
