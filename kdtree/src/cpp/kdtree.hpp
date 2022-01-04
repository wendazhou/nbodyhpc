#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <span.hpp>

namespace wenda {
namespace kdtree {

//! This structure encapsulates information to compute the l2 distance between two points.
struct L2Distance {
    //! Computes the distance between two points.
    template <typename T, size_t R>
    T operator()(std::array<T, R> const &left, std::array<T, R> const &right) const {
        T result = 0;

        for (size_t i = 0; i < R; ++i) {
            result += (left[i] - right[i]) * (left[i] - right[i]);
        }

        return result;
    }

    //! Computes the distance from a point to an axis-aligned box
    template <typename T, size_t R>
    T box_distance(std::array<T, R> const &point, std::array<T, 2 * R> const &box) const {
        T result = 0;

        for (size_t i = 0; i < R; ++i) {
            auto delta_left = std::max(box[2 * i] - point[i], T{0});
            auto delta_right = std::max(point[i] - box[2 * i + 1], T{0});
            result += delta_left * delta_left + delta_right * delta_right;
        }

        return result;
    }

    //! Post-processes the result of the distance computation.
    template <typename T> T postprocess(T value) const { return std::sqrt(value); }

    //! Returns a box with no constraints of the given dimension.
    template <typename T, size_t R>
    std::array<T, 2 * R> initial_box(std::array<T, R> const &) const {
        std::array<T, 2 * R> result;

        for (size_t i = 0; i < R; ++i) {
            result[2 * i] = std::numeric_limits<T>::lowest();
            result[2 * i + 1] = std::numeric_limits<T>::max();
        }

        return result;
    }
};

//! This structure encapsulates information to compute l2 distance with periodic boundary conditions.
template <typename T> struct L2PeriodicDistance {
    //! Size of the box for periodic boundary conditions.
    T box_size_;

    //! Computes the distance between two points.
    template <size_t R>
    T operator()(std::array<T, R> const &left, std::array<T, R> const &right) const {
        T result = 0;

        for (size_t i = 0; i < R; ++i) {
            auto delta = left[i] - right[i];
            auto delta_p = delta + box_size_;
            auto delta_m = delta - box_size_;

            result += std::min({delta * delta, delta_p * delta_p, delta_m * delta_m});
        }

        return result;
    }

    //! Computes the distance from a point to an axis-aligned box which does not cross the periodic boundary.
    template <size_t R>
    T box_distance(std::array<T, R> const &point, std::array<T, 2 * R> const &box) const {
        T result = 0;

        for (size_t i = 0; i < R; ++i) {
            if (point[i] < box[2 * i]) {
                auto delta = box[2 * i] - point[i];
                auto delta_wrap = point[i] + box_size_ - box[2 * i + 1];
                auto delta_min = std::min(delta, delta_wrap);
                result += delta_min * delta_min;
            } else if (point[i] > box[2 * i + 1]) {
                auto delta = point[i] - box[2 * i + 1];
                auto delta_wrap = box[2 * i] + box_size_ - point[i];
                auto delta_min = std::min(delta, delta_wrap);
                result += delta_min * delta_min;
            }
        }

        return result;
    }

    T postprocess(T value) const { return std::sqrt(value); }

    template <size_t R> std::array<T, 2 * R> initial_box(std::array<T, R> const &) const {
        std::array<T, 2 * R> result;

        for (size_t i = 0; i < R; ++i) {
            result[2 * i] = 0;
            result[2 * i + 1] = box_size_;
        }

        return result;
    }
};

//! Statistics of queries into a kd-tree.
struct KDTreeQueryStatistics {
    //! Number of nodes visited during the query.
    size_t nodes_visited;
    //! Number of nodes pruned during the query.
    size_t nodes_pruned;
    //! Number of points for which distance was evaluated during the query.
    size_t points_visited;
};

//! Configuration for building a kd-tree
struct KDTreeConfiguration {
    //! Number of points in leaf nodes (where we switch to brute-force)
    int leaf_size = 64;
    //! Maximum number of threads to use during tree construction. If -1, use all available threads.
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

  public:
    /** Builds a new KD-tree from the given positions.
     *
     * @param positions The positions to build the tree from.
     * @param config Configuration to use when building the tree.
     */
    KDTree(tcb::span<const std::array<float, 3>> positions, KDTreeConfiguration const &config = {});

    /** Builds a new KD-tree from the given positions.
     * 
     * This constructor makes use of a pre-allocated positions vector.
     * To build from an array of positions, see other constructor.
     * 
     * @param positions_and_indices The positions and indices to build the tree from.
     * @param config Configuration to use when building the tree.
     * 
     */
    KDTree(std::vector<PositionAndIndex>&& positions_and_indices, KDTreeConfiguration const &config = {});

    KDTree(KDTree const &) = default;
    KDTree(KDTree &&) noexcept = default;

    tcb::span<const KDTreeNode> nodes() const noexcept { return nodes_; }
    tcb::span<const PositionAndIndex> positions() const noexcept { return positions_; }

    /** Searches the tree for the nearest neighbors of the given query point.
     *
     * @param position The point at which to query
     * @param k The number of nearest neighbors to return
     * @param statistics[out] Optional pointer to a KDTreeQueryStatistics structure to store
     * statistics about the query.
     *
     * @returns A vector of the nearest neighbors, sorted by distance. Each neighbor is represented
     * by a pair of a distance and an index corresponding to the index of the point in the original
     * positions array used to construct the tree.
     *
     */
    template <typename Distance>
    std::vector<std::pair<float, uint32_t>> find_closest(
        std::array<float, 3> const &position, size_t k, Distance const &distance = {},
        KDTreeQueryStatistics *statistics = nullptr) const;
};

extern template std::vector<std::pair<float, uint32_t>> KDTree::find_closest<L2Distance>(
    std::array<float, 3> const &position, size_t k, L2Distance const &,
    KDTreeQueryStatistics *statistics) const;

extern template std::vector<std::pair<float, uint32_t>> KDTree::find_closest<L2PeriodicDistance<float>>(
    std::array<float, 3> const &position, size_t k, L2PeriodicDistance<float> const &,
    KDTreeQueryStatistics *statistics) const;

} // namespace kdtree
} // namespace wenda
