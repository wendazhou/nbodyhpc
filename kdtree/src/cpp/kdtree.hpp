#pragma once

#include <array>
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


} // namespace kdtree
} // namespace wenda
