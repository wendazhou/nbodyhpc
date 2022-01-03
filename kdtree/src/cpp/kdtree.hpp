#pragma once

#include <array>
#include <limits>
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
float query_kdtree(KDTreeNode *tree, std::array<float, 3> const& query, float distance_bound = std::numeric_limits<float>::max());

}
}
