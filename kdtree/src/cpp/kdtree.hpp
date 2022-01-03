#pragma once

#include <array>
#include <memory>
#include <vector>

#include "span.hpp"

namespace wenda {
namespace kdtree {

struct KDTreeNode {
    int dimension_;
    float split_;
    bool leaf_;

    union {
        struct {
            KDTreeNode *left_;
            KDTreeNode *right_;
        } children_;

        std::vector<std::array<float, 3>> positions_;
    };

    KDTreeNode(
        int dimension, float split, std::unique_ptr<KDTreeNode> &&left,
        std::unique_ptr<KDTreeNode> &&right)
        : dimension_(dimension), split_(split),
          leaf_(false), children_{left.release(), right.release()} {}

    KDTreeNode(int dimension, std::vector<std::array<float, 3>> positions)
        : dimension_(dimension), split_(0), leaf_(true), positions_(std::move(positions)) {}

    KDTreeNode(KDTreeNode const &) = delete;

    ~KDTreeNode() noexcept {
        if (!leaf_) {
            delete children_.left_;
            delete children_.right_;
        } else {
            positions_.~vector();
        }
    }
};

std::unique_ptr<KDTreeNode> build_kdtree(tcb::span<std::array<float, 3>> positions);

/** Query the kdtree for the nearest neighbors of a given point.
 * 
 * @param tree A pointer to the root of the kdtree to query.
 * @param point The point to query for.
 * @param k The number of nearest neighbors to return.
 * @param max_distance The maximum distance of the neighbors to consider.
 * 
 */
std::vector<float> query_kdtree_knn(
    KDTreeNode *tree, std::array<float, 3> const &query, int k = 1);

} // namespace kdtree
} // namespace wenda
