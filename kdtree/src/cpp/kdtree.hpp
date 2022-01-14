#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

#include <span.hpp>

namespace wenda {
namespace kdtree {

#ifdef _MSC_VER
inline void *aligned_alloc(size_t alignment, size_t size) noexcept {
    return _aligned_malloc(size, alignment);
}

inline void aligned_free(void *ptr) noexcept { _aligned_free(ptr); }
#else
inline void *aligned_alloc(size_t alignment, size_t size) noexcept {
    // round up size to next multiple of alignment
    size = (size + alignment - 1) & ~(alignment - 1);
    // For some reason std::aligned_alloc not exposed in libstdc++ on MacOS
    return ::aligned_alloc(alignment, size);
}

inline void aligned_free(void *ptr) noexcept { return free(ptr); }
#endif

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

//! This structure encapsulates information to compute l2 distance with periodic boundary
//! conditions.
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

    //! Computes the distance from a point to an axis-aligned box which does not cross the periodic
    //! boundary.
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
    //! All leaf nodes must have a number of points which is divisible by this quantity.
    int block_size = 8;
};

struct alignas(16) PositionAndIndex {
    std::array<float, 3> position;
    uint32_t index;
};

template <typename T> struct OffsetRangeContainerWrapper {
    T &container_;
    size_t offset;
    size_t count;

    OffsetRangeContainerWrapper(T &container)
        : container_(container), offset(0), count(container.size()) {}
    OffsetRangeContainerWrapper(T &container, size_t offset, size_t count)
        : container_(container), offset(offset), count(count) {}

    decltype(auto) begin() { return container_.begin() + offset; }
    decltype(auto) begin() const { return container_.begin() + offset; }
    decltype(auto) end() { return container_.begin() + offset + count; }
    decltype(auto) end() const { return container_.begin() + offset + count; }

    decltype(auto) operator[](size_t i) { return container_[offset + i]; }
    decltype(auto) operator[](size_t i) const { return container_[offset + i]; }

    size_t size() const { return count; }
};

namespace detail {
template <typename ArrayBase, typename Derived> struct PositionAndIndexIteratorBase {
    typedef PositionAndIndex value_type;
    typedef std::random_access_iterator_tag iterator_category;
    typedef std::ptrdiff_t difference_type;

    ptrdiff_t offset_;
    ArrayBase array_;

    PositionAndIndexIteratorBase() = default;
    PositionAndIndexIteratorBase(ArrayBase array, ptrdiff_t offset)
        : offset_(offset), array_(array) {}
    PositionAndIndexIteratorBase(PositionAndIndexIteratorBase const &) = default;

    Derived &operator++() {
        ++offset_;
        return *static_cast<Derived *>(this);
    }

    Derived operator++(int) {
        Derived result = *static_cast<Derived *>(this);
        ++offset_;
        return result;
    }

    Derived &operator--() {
        --offset_;
        return *static_cast<Derived *>(this);
    }

    Derived operator--(int) {
        Derived result = *static_cast<Derived *>(this);
        --offset_;
        return result;
    }

    Derived &operator+=(ptrdiff_t n) {
        offset_ += n;
        return *static_cast<Derived *>(this);
    }

    Derived &operator-=(ptrdiff_t n) {
        offset_ -= n;
        return *static_cast<Derived *>(this);
    }

    decltype(auto) operator[](ptrdiff_t n) { return *(*static_cast<Derived *>(this) + n); }

    Derived operator+(ptrdiff_t n) const { return Derived(array_, offset_ + n); }

    Derived operator-(ptrdiff_t n) const { return Derived(array_, offset_ - n); }

    ptrdiff_t operator-(Derived const &other) const { return offset_ - other.offset_; }

    bool operator==(Derived const &other) const { return offset_ == other.offset_; }

    bool operator!=(Derived const &other) const { return offset_ != other.offset_; }

    bool operator<(Derived const &other) const { return offset_ < other.offset_; }

    bool operator<=(Derived const &other) const { return offset_ <= other.offset_; }

    bool operator>(Derived const &other) const { return offset_ > other.offset_; }

    bool operator>=(Derived const &other) const { return offset_ >= other.offset_; }

    Derived &operator=(Derived const &other) {
        offset_ = other.offset_;
        return *static_cast<Derived *>(this);
    }
};

template <size_t R, typename T, typename IndexT> struct PositionAndIndexIterator;

template <size_t R, typename T, typename IndexT> struct ConstPositionAndIndexIterator;

} // namespace detail

template <size_t R = 3, typename T = float, typename IndexT = uint32_t>
struct PositionAndIndexArray {
    struct PositionAndIndexProxy {
        std::array<std::reference_wrapper<float>, R> position;
        uint32_t &index;

        operator PositionAndIndex() const {
            return {std::array<float, 3>{position[0], position[1], position[2]}, index};
        }
    };

    typedef detail::PositionAndIndexIterator<R, T, IndexT> iterator;
    typedef detail::ConstPositionAndIndexIterator<R, T, IndexT> const_iterator;

    std::array<T *, R> positions_;
    std::vector<IndexT> indices_;

    PositionAndIndexArray() = default;
    PositionAndIndexArray(PositionAndIndexArray const &) = delete;
    PositionAndIndexArray(PositionAndIndexArray &&other) noexcept
        : positions_(std::move(other.positions_)), indices_(std::move(other.indices_)) {
        std::fill(other.positions_.begin(), other.positions_.end(), nullptr);
    }

    PositionAndIndexArray &operator=(PositionAndIndexArray const &) = delete;
    PositionAndIndexArray &operator=(PositionAndIndexArray &&other) noexcept {
        std::swap(positions_, other.positions_);
        std::swap(indices_, other.indices_);

        return *this;
    }

    explicit PositionAndIndexArray(size_t n) : indices_(n) {
        for (size_t i = 0; i < R; ++i) {
            positions_[i] = static_cast<T *>(aligned_alloc(64, sizeof(T) * n));
        }
    }

    template <
        typename Container,
        typename std::enable_if<std::negation<std::is_integral<Container>>::value, bool>::type = true>
    explicit PositionAndIndexArray(Container const &positions)
        : PositionAndIndexArray(std::size(positions)) {
        using std::begin;
        using std::end;

        for (size_t i = 0; i < R; ++i) {
            std::transform(
                begin(positions), end(positions), positions_[i], [i](PositionAndIndex const &p) {
                    return p.position[i];
                });
        }

        std::transform(
            begin(positions), end(positions), indices_.begin(), [](PositionAndIndex const &p) {
                return p.index;
            });
    }

    ~PositionAndIndexArray() noexcept {
        for (auto &position_ptr : positions_) {
            aligned_free(position_ptr);
            position_ptr = nullptr;
        }
    }

    size_t size() const { return indices_.size(); }

    PositionAndIndexProxy operator[](size_t i) {
        return PositionAndIndexProxy{
            {positions_[0][i], positions_[1][i], positions_[2][i]}, indices_[i]};
    }

    PositionAndIndex operator[](size_t i) const {
        return PositionAndIndex{
            {positions_[0][i], positions_[1][i], positions_[2][i]}, indices_[i]};
    }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size()); }

    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size()); }
};

namespace detail {

template <size_t R, typename T, typename IndexT>
struct PositionAndIndexIterator
    : detail::PositionAndIndexIteratorBase<
          PositionAndIndexArray<R, T, IndexT> *, PositionAndIndexIterator<R, T, IndexT>> {
    typedef PositionAndIndexArray<R, T, IndexT> Array;
    typedef detail::PositionAndIndexIteratorBase<Array *, PositionAndIndexIterator> iterator_base;
    typedef typename iterator_base::value_type value_type;
    typedef typename iterator_base::difference_type difference_type;
    typedef typename iterator_base::iterator_category iterator_category;
    typedef typename Array::PositionAndIndexProxy reference;
    typedef void pointer;

    PositionAndIndexIterator() = default;
    PositionAndIndexIterator(Array *array, size_t offset) : iterator_base(array, offset) {}
    PositionAndIndexIterator(PositionAndIndexIterator const &) = default;

    using iterator_base::operator==;
    using iterator_base::operator!=;

    reference operator*() const { return (*this->array_)[this->offset_]; }
};

template <size_t R, typename T, typename IndexT>
struct ConstPositionAndIndexIterator : detail::PositionAndIndexIteratorBase<
                                           PositionAndIndexArray<R, T, IndexT> const *,
                                           ConstPositionAndIndexIterator<R, T, IndexT>> {

    typedef PositionAndIndexArray<R, T, IndexT> Array;

    typedef detail::PositionAndIndexIteratorBase<Array const *, ConstPositionAndIndexIterator>
        iterator_base;
    typedef typename iterator_base::value_type value_type;
    typedef typename iterator_base::difference_type difference_type;
    typedef typename iterator_base::iterator_category iterator_category;
    typedef typename Array::PositionAndIndexProxy reference;
    typedef void pointer;

    ConstPositionAndIndexIterator() = default;
    ConstPositionAndIndexIterator(Array const *array, size_t offset)
        : iterator_base(array, offset) {}
    ConstPositionAndIndexIterator(ConstPositionAndIndexIterator const &) = default;

    using iterator_base::operator==;
    using iterator_base::operator!=;

    value_type operator*() const { return (*this->array_)[this->offset_]; }
};

template <size_t R, typename T, typename IndexT>
void iter_swap(
    PositionAndIndexIterator<R, T, IndexT> const &it1,
    PositionAndIndexIterator<R, T, IndexT> const &it2) noexcept {
    for (size_t i = 0; i < R; ++i) {
        std::swap(it1.array_->positions_[i][it1.offset_], it2.array_->positions_[i][it2.offset_]);
    }
    std::swap(it1.array_->indices_[it1.offset_], it2.array_->indices_[it2.offset_]);
}

} // namespace detail

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
    PositionAndIndexArray<3, float> positions_;

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
    KDTree(
        std::vector<PositionAndIndex> &&positions_and_indices,
        KDTreeConfiguration const &config = {});

    KDTree(KDTree &&) noexcept = default;

    tcb::span<const KDTreeNode> nodes() const noexcept { return nodes_; }
    PositionAndIndexArray<3> const &positions() const noexcept { return positions_; }

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

extern template std::vector<std::pair<float, uint32_t>>
KDTree::find_closest<L2PeriodicDistance<float>>(
    std::array<float, 3> const &position, size_t k, L2PeriodicDistance<float> const &,
    KDTreeQueryStatistics *statistics) const;

} // namespace kdtree
} // namespace wenda
