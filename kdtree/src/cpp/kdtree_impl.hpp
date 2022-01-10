#pragma once

#include <future>
#include <mutex>

#include "kdtree.hpp"
#include "kdtree_opt.hpp"
#include "tournament_tree.hpp"

namespace wenda {

namespace kdtree {

namespace detail {

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

//! Utility structure used to store state of KD-tree search.
template <
    typename DistanceT = L2Distance,
    typename QueueT = PriorityQueue<std::pair<float, uint32_t>, PairLessFirst>,
    template <typename, typename> typename InserterT = InsertShorterDistanceVanilla>
struct KDTreeQuery {
    typedef std::pair<float, uint32_t> result_t;
    typedef QueueT queue_t;

    DistanceT const &distance_;
    tcb::span<const PositionAndIndex> positions_;
    tcb::span<const KDTree::KDTreeNode> nodes_;
    std::array<float, 3> const &query_;
    QueueT distances_;
    size_t num_nodes_visited = 0;
    size_t num_nodes_pruned = 0;
    size_t num_points_visited = 0;

    KDTreeQuery(
        KDTree const &tree, DistanceT const &distance, std::array<float, 3> const &query, size_t k)
        : distance_(distance), positions_(tree.positions()), nodes_(tree.nodes()), query_(query),
          distances_(k, {std::numeric_limits<float>::max(), -1}) {}

    void process_leaf(KDTree::KDTreeNode const &node) {
        uint32_t const num_points = node.right_ - node.left_;
        auto node_positions = positions_.subspan(node.left_, num_points);

        InserterT<DistanceT, QueueT> insert_shorter_distance;
        insert_shorter_distance(node_positions, query_, distances_, distance_);

        num_points_visited += node_positions.size();
    }

    void compute(KDTree::KDTreeNode const* node) {
        return compute(node, distance_.initial_box(query_));
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

} // namespace detail

} // namespace kdtree

} // namespace wenda