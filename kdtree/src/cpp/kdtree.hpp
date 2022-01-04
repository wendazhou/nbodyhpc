#pragma once

#include <array>
#include <vector>

#include <span.hpp>

namespace wenda {
namespace kdtree {

struct KDTreeQueryStatistics {
    size_t nodes_visited;
    size_t nodes_pruned;
    size_t points_visited;
};

struct KDTreeConfiguration {
    int leaf_size = 8;
    int max_threads = 0;
};

struct PositionAndIndex {
    std::array<float, 3> position;
    uint32_t index;
};

class KDTree {
  public:
    /** This structure represents a single node in the KD-tree
     *
     */
    struct KDTreeNode {
        //! The dimension at which the split occurs.
        //! This field additionally indicates that the node is a leaf node
        //! when the dimension is specified to be -1.
        int dimension_;
        //! The value at which the split is performed, if not a leaf node.
        float split_;

        //! For leaf nodes, the start index of the positions in this node.
        //! For non-leaf nodes, the index of the left child node.
        uint32_t left_;
        //! For leaf nodes, the end index of the positions in this node.
        //! For non-leaf nodes, the index of the right child node.
        uint32_t right_;
    };

  private:
    std::vector<KDTreeNode> nodes_;
    std::vector<PositionAndIndex> positions_;
    uint32_t max_leaf_size_;

  public:
    /** Builds a new KD-tree from the given positions.
     *
     * @param positions The positions to build the tree from.
     * @param multithreaded If True, indicates that multithreading should be used for construction.
     */
    KDTree(tcb::span<const std::array<float, 3>> positions, KDTreeConfiguration const& config = {});
    KDTree(KDTree const &) = default;
    KDTree(KDTree &&) noexcept = default;

    tcb::span<const KDTreeNode> nodes() const noexcept { return nodes_; }
    tcb::span<const PositionAndIndex> positions() const noexcept { return positions_; }

    /** Searches the tree for the nearest neighbors of the given query point.
     * 
     * @param position The point at which to query
     * @param k The number of nearest neighbors to return
     * @param statistics[out] Optional pointer to a KDTreeQueryStatistics structure to store statistics about the query.
     * 
     * @returns A vector of the nearest neighbors, sorted by distance. Each neighbor is represented by a pair of a distance and an index
     *          corresponding to the index of the point in the original positions array used to construct the tree.
     * 
     */
    std::vector<std::pair<float, uint32_t>> find_closest(
        std::array<float, 3> const &position, size_t k,
        KDTreeQueryStatistics *statistics = nullptr) const;
};

} // namespace kdtree
} // namespace wenda
