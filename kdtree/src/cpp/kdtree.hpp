#pragma once

#include <array>
#include <memory>
#include <vector>

#include "span.hpp"

namespace wenda {
namespace kdtree {

struct KDTreeQueryStatistics {
    size_t nodes_visited;
    size_t nodes_pruned;
    size_t points_visited;
};

class KDTree {
  public:
    /** This structure represents a single node in the KD-tree
     *
     */
    struct KDTreeNode {
        //! The dimension at which the split occurs
        int dimension_;
        //! The value at which the split is performed
        float split_;
        //! Indicates whether this is a leaf node
        bool leaf_;

        //! For leaf nodes, the start index of the positions in this node.
        //! For non-leaf nodes, the index of the left child node.
        uint32_t left_;
        //! For leaf nodes, the end index of the positions in this node.
        //! For non-leaf nodes, the index of the right child node.
        uint32_t right_;
    };

  private:
    std::vector<KDTreeNode> nodes_;
    std::vector<std::array<float, 3>> positions_;

  public:
    KDTree(tcb::span<const std::array<float, 3>> positions);

    tcb::span<const KDTreeNode> nodes() const noexcept { return nodes_; }
    tcb::span<const std::array<float, 3>> positions() const noexcept { return positions_; }

    std::vector<float> find_closest(
        std::array<float, 3> const &position, size_t k,
        KDTreeQueryStatistics *statistics = nullptr) const;
};

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

std::unique_ptr<KDTreeNode> build_kdtree(tcb::span<const std::array<float, 3>> positions);

/** Query the kdtree for the nearest neighbors of a given point.
 *
 * @param tree A pointer to the root of the kdtree to query.
 * @param point The point to query for.
 * @param k The number of nearest neighbors to return.
 * @param max_distance The maximum distance of the neighbors to consider.
 * @param[out,optional] If not null, the statistics of the query will be stored here.
 *
 */
std::vector<float> query_kdtree_knn(
    KDTreeNode const *tree, std::array<float, 3> const &query, int k = 1,
    KDTreeQueryStatistics *statistics = nullptr);

} // namespace kdtree
} // namespace wenda
