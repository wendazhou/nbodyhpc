#include "kdtree.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <vector>

#include "span.hpp"

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
        tcb::span<int>(indices.begin(), median_it - 1), positions, (dimension + 1) % 3);
    auto right_tree = build_kdtree_implementation(
        tcb::span<int>(median_it + 1, indices.end()), positions, (dimension + 1) % 3);

    return std::make_unique<KDTreeNode>(
        dimension, split, std::move(left_tree), std::move(right_tree));
}

std::unique_ptr<KDTreeNode> build_kdtree(tcb::span<std::array<float, 3>> positions) {
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

float query_kdtree(KDTreeNode *tree, std::array<float, 3> const& query, float distance_bound) {
    if (tree->leaf_) {
        // return distance to closest point in tree leaf

        return std::transform_reduce(
            tree->positions_.begin(),
            tree->positions_.end(),
            std::numeric_limits<float>::max(),
            [](float a, float b) { return std::min(a, b); },
            [&](std::array<float, 3> const &x) { return l2_distance(x, query); });
    }

    KDTreeNode* closer, * further;

    if (query[tree->dimension_] < tree->split_) {
        closer = tree->children_.left_;
        further = tree->children_.right_;
    } else {
        closer = tree->children_.right_;
        further = tree->children_.left_;
    }

    float distance_closer = query_kdtree(closer, query, distance_bound);

    distance_bound = std::min(distance_bound, distance_closer);
    auto delta_dimension = query[tree->dimension_] - tree->split_;

    // prune evaluation of further node if it is guaranteed to be further.
    if (distance_bound < delta_dimension * delta_dimension) {
        return distance_bound;
    }

    float distance_further = query_kdtree(further, query, distance_bound);

    return std::min(distance_bound, distance_further);
}

} // namespace kdtree

} // namespace wenda
