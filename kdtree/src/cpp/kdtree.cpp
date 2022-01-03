#include "kdtree.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

#include "span.hpp"
#include "utils.hpp"

namespace wenda {

namespace kdtree {

std::unique_ptr<KDTreeNode> build_kdtree_implementation(
    tcb::span<int> indices, tcb::span<const std::array<float, 3>> positions, int dimension) {
    if (indices.size() < 8) {
        // leaf node
        std::vector<std::array<float, 3>> positions_copy(indices.size());
        std::transform(
            indices.begin(), indices.end(), positions_copy.begin(), [&positions](int index) {
                return positions[index];
            });

        return std::make_unique<KDTreeNode>(0, std::move(positions_copy));
    }

    // split along median
    auto median_it = indices.begin() + indices.size() / 2;

    std::nth_element(
        indices.begin(), median_it, indices.end(), [&positions, dimension](int a, int b) {
            return positions[a][dimension] < positions[b][dimension];
        });

    float split = positions[*median_it][dimension];

    // recurse
    auto left_tree = build_kdtree_implementation(
        tcb::span<int>(indices.begin(), median_it), positions, (dimension + 1) % 3);
    auto right_tree = build_kdtree_implementation(
        tcb::span<int>(median_it, indices.end()), positions, (dimension + 1) % 3);

    return std::make_unique<KDTreeNode>(
        dimension, split, std::move(left_tree), std::move(right_tree));
}

std::unique_ptr<KDTreeNode> build_kdtree(tcb::span<const std::array<float, 3>> positions) {
    std::vector<int> indices(positions.size());
    std::iota(indices.begin(), indices.end(), 0);

    return build_kdtree_implementation(indices, positions, 0);
}

template <typename T, size_t R>
T l2_distance(std::array<T, R> const &left, std::array<T, R> const &right) {
    T result = 0;

    for (size_t i = 0; i < R; ++i) {
        result += (left[i] - right[i]) * (left[i] - right[i]);
    }

    return result;
}

namespace {

uint32_t build_kdtree_offsets(
    std::vector<KDTree::KDTreeNode> &nodes, tcb::span<std::array<float, 3>> positions,
    int dimension, uint32_t left) {
    if (positions.size() < 8) {
        nodes.emplace_back(
            dimension, 0.0f, true, left, static_cast<uint32_t>(left + positions.size()));
        return static_cast<uint32_t>(nodes.size() - 1);
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

    uint32_t current_idx = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back(dimension, split, false, 0, 0);

    uint32_t left_idx = build_kdtree_offsets(
        nodes,
        tcb::span<std::array<float, 3>>(positions.begin(), median_it),
        (dimension + 1) % 3,
        left);
    uint32_t right_idx = build_kdtree_offsets(
        nodes,
        tcb::span<std::array<float, 3>>(median_it, positions.end()),
        (dimension + 1) % 3,
        left + static_cast<uint32_t>(std::distance(positions.begin(), median_it)));

    nodes[current_idx].left_ = left_idx;
    nodes[current_idx].right_ = right_idx;

    return current_idx;
}

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

    void compute(uint32_t node_idx) {
        auto const &node = nodes_[node_idx];

        num_nodes_visited += 1;

        if (node.leaf_) {
            tcb::span<const std::array<float, 3>> node_positions(
                positions_.data() + node.left_, positions_.data() + node.right_);

            for (auto const &p : node_positions) {
                push_point(p);
            }

            num_points_visited += node_positions.size();

            return;
        }

        uint32_t closer, further;

        if (query_[node.dimension_] < node.split_) {
            closer = node.left_;
            further = node.right_;
        } else {
            closer = node.right_;
            further = node.left_;
        }

        compute(closer);

        auto delta_dimension = query_[node.dimension_] - node.split_;

        if (distances_.top() < delta_dimension * delta_dimension) {
            num_nodes_pruned += 1;
            return;
        }

        compute(further);
    }
};

} // namespace

KDTree::KDTree(tcb::span<const std::array<float, 3>> positions)
    : positions_(positions.begin(), positions.end()) {
    build_kdtree_offsets(nodes_, positions_, 0, 0);
}

std::vector<float> KDTree::find_closest(
    std::array<float, 3> const &position, size_t k, KDTreeQueryStatistics *statistics) const {

    KDTreeQuery query(*this, position, k);
    query.compute(0);

    if (statistics) {
        statistics->nodes_visited = query.num_nodes_visited;
        statistics->nodes_pruned = query.num_nodes_pruned;
        statistics->points_visited = query.num_points_visited;
    }

    std::vector result(std::move(get_container_from_adapter(query.distances_)));
    std::sort(result.begin(), result.end());
    return result;
}

namespace {

struct KDNodeQuery {
    std::array<float, 3> const &query_;
    std::priority_queue<float> distances_;
    size_t num_nodes_visited = 0;
    size_t num_nodes_pruned = 0;
    size_t num_points_visited = 0;

    KDNodeQuery(std::array<float, 3> const &query, int k)
        : query_(query),
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

    void compute(KDTreeNode const *tree) {
        num_nodes_visited += 1;

        if (tree->leaf_) {
            for (auto const &p : tree->positions_) {
                push_point(p);
            }

            num_points_visited += tree->positions_.size();

            return;
        }

        KDTreeNode const *closer, *further;

        if (query_[tree->dimension_] < tree->split_) {
            closer = tree->children_.left_;
            further = tree->children_.right_;
        } else {
            closer = tree->children_.right_;
            further = tree->children_.left_;
        }

        compute(closer);

        auto delta_dimension = query_[tree->dimension_] - tree->split_;

        if (distances_.top() < delta_dimension * delta_dimension) {
            num_nodes_pruned += 1;
            return;
        }

        compute(further);
    }
};

} // namespace

std::vector<float> query_kdtree_knn(
    KDTreeNode const *tree, std::array<float, 3> const &query, int k,
    KDTreeQueryStatistics *statistics) {
    KDNodeQuery q(query, k);
    q.compute(tree);

    if (statistics) {
        statistics->nodes_visited = q.num_nodes_visited;
        statistics->nodes_pruned = q.num_nodes_pruned;
        statistics->points_visited = q.num_points_visited;
    }

    auto result = std::move(get_container_from_adapter(q.distances_));
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace kdtree

} // namespace wenda
