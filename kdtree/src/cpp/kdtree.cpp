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
        nodes.push_back({-1, 0.0f, left, static_cast<uint32_t>(left + positions.size())});
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
    nodes.push_back({dimension, split, 0u, 0u});

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

    void compute(KDTree::KDTreeNode const* node) {
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

KDTree::KDTree(tcb::span<const std::array<float, 3>> positions)
    : positions_(positions.begin(), positions.end()) {
    build_kdtree_offsets(nodes_, positions_, 0, 0);
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
