#pragma once

#include <array>
#include <cstdlib>
#include <iterator>
#include <vector>

//! This file contains an implementation to represent an array of n-dimensional
//! points in a structure-of-array (SoA) format.
//! This is used over the more intuitive array-of-structures (e.g. std::vector<std::array<float,
//! 3>>) due to performance reasons.

namespace wenda {

namespace kdtree {

void *aligned_alloc(size_t alignment, size_t size) noexcept;
void aligned_free(void *ptr) noexcept;

template <size_t R = 3, typename T = float> struct PositionAndIndex {
    std::array<T, R> position;
    uint32_t index;
};

//! Simple wrapper around a container which represents a sub-span of that container.
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

//! Base class implementing common operations for PositionAndIndexIterator and its const version.
template <typename ArrayBase, typename Derived> struct PositionAndIndexIteratorBase {
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

    decltype(auto) operator[](ptrdiff_t n) const {
        return *(*static_cast<const Derived *>(this) + n);
    }

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

//! Proxy class for a PositionAndIndex instance.
//! As we use a SoA representation, this proxy is required to expose PositionAndIndexArray
//! as a range of PositionAndIndex instances.
template <size_t R, typename T, typename IndexT> struct PositionAndIndexProxy {
    std::array<std::reference_wrapper<T>, R> position;
    uint32_t &index;

    operator PositionAndIndex<R, T>() const noexcept {
        std::array<T, R> positions;

        for (size_t i = 0; i < R; ++i) {
            positions[i] = position[i].get();
        }

        return {positions, index};
    }

    const PositionAndIndexProxy &operator=(PositionAndIndex<R, T> const &other) const noexcept {
        for (size_t i = 0; i < R; ++i) {
            position[i].get() = other.position[i];
        }
        index = other.index;
        return *this;
    }

    const PositionAndIndexProxy &operator=(PositionAndIndexProxy const &other) const noexcept {
        for (size_t i = 0; i < R; ++i) {
            position[i].get() = other.position[i];
        }
        index = other.index;
        return *this;
    }
};

} // namespace detail

//! This class represents a sequence of PositionAndIndex instances.
//! It encapsulates all points in a structure-of-array (SoA) format for efficiency.
//! For convenience, it also exposes the points it contains as a range of PositionAndIndex
//! through a set of accessors and proxies.
template <size_t R = 3, typename T = float, typename IndexT = uint32_t>
struct PositionAndIndexArray {
    static const size_t dimension = R;
    typedef T element_type;

    typedef PositionAndIndex<R, T> value_type;

    typedef detail::PositionAndIndexIterator<R, T, IndexT> iterator;
    typedef detail::ConstPositionAndIndexIterator<R, T, IndexT> const_iterator;
    typedef detail::PositionAndIndexProxy<R, T, IndexT> PositionAndIndexProxy;

    std::array<T *, R> positions_;
    std::vector<IndexT> indices_;

    //! Constructs an empty array.
    PositionAndIndexArray() = default;

    PositionAndIndexArray(PositionAndIndexArray const &other) : indices_(other.indices_) {
        for (size_t i = 0; i < R; ++i) {
            positions_[i] = static_cast<T *>(aligned_alloc(64, sizeof(T) * indices_.size()));
            std::copy(other.positions_[i], other.positions_[i] + indices_.size(), positions_[i]);
        }
    }

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

    //! Constructs an array with the given capacity.
    explicit PositionAndIndexArray(size_t n) : indices_(n) {
        for (size_t i = 0; i < R; ++i) {
            positions_[i] = static_cast<T *>(aligned_alloc(64, sizeof(T) * n));
        }
    }

    //! Constructs an array from a given range.
    template <
        typename Container,
        typename std::enable_if<std::negation<std::is_integral<Container>>::value, bool>::type =
            true>
    explicit PositionAndIndexArray(Container const &positions)
        : PositionAndIndexArray(std::size(positions)) {
        using std::begin;
        using std::end;

        for (size_t i = 0; i < R; ++i) {
            std::transform(
                begin(positions), end(positions), positions_[i], [i](PositionAndIndex<R, T> const &p) {
                    return p.position[i];
                });
        }

        std::transform(
            begin(positions), end(positions), indices_.begin(), [](PositionAndIndex<R, T> const &p) {
                return p.index;
            });
    }

    ~PositionAndIndexArray() noexcept {
        for (auto &position_ptr : positions_) {
            aligned_free(position_ptr);
            position_ptr = nullptr;
        }
    }

    //! Obtain the current size of the array
    size_t size() const noexcept { return indices_.size(); }

    PositionAndIndexProxy operator[](size_t i) noexcept {
        return PositionAndIndexProxy{
            {positions_[0][i], positions_[1][i], positions_[2][i]}, indices_[i]};
    }

    PositionAndIndex<R, T> operator[](size_t i) const noexcept {
        std::array<T, R> pos;

        for (size_t dim = 0; dim < R; ++dim) {
            pos[dim] = positions_[dim][i];
        }

        return PositionAndIndex{pos, indices_[i]};
    }

    iterator begin() noexcept { return iterator(this, 0); }
    iterator end() noexcept { return iterator(this, size()); }

    const_iterator begin() const noexcept { return const_iterator(this, 0); }
    const_iterator end() const noexcept { return const_iterator(this, size()); }

    //! Swaps the elements at index i and j.
    void swap_elements(size_t i, size_t j) noexcept {
        std::swap(indices_[i], indices_[j]);
        for (size_t k = 0; k < R; ++k) {
            std::swap(positions_[k][i], positions_[k][j]);
        }
    }
};

namespace detail {

//! Iterator for PositionAndIndexArray.
//! Note that this iterator does not properly model the legacy random iterator concept (as it is a
//! proxy iterator). However, it correctly supports C++20 new range random iterator concepts.
template <size_t R, typename T, typename IndexT>
struct PositionAndIndexIterator
    : detail::PositionAndIndexIteratorBase<
          PositionAndIndexArray<R, T, IndexT> *, PositionAndIndexIterator<R, T, IndexT>> {
    typedef PositionAndIndexArray<R, T, IndexT> Array;
    typedef detail::PositionAndIndexIteratorBase<Array *, PositionAndIndexIterator> iterator_base;
    typedef PositionAndIndex<R, T> value_type;
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
PositionAndIndexIterator<R, T, IndexT>
operator+(ptrdiff_t n, PositionAndIndexIterator<R, T, IndexT> const &it) {
    return it + n;
}

template <size_t R, typename T, typename IndexT>
struct ConstPositionAndIndexIterator : detail::PositionAndIndexIteratorBase<
                                           PositionAndIndexArray<R, T, IndexT> const *,
                                           ConstPositionAndIndexIterator<R, T, IndexT>> {

    typedef PositionAndIndexArray<R, T, IndexT> Array;

    typedef detail::PositionAndIndexIteratorBase<Array const *, ConstPositionAndIndexIterator>
        iterator_base;
    typedef PositionAndIndex<R, T> value_type;
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
    it1.array_->swap_elements(it1.offset_, it2.offset_);
}

template <size_t R, typename T, typename IndexT>
PositionAndIndex<R, T> iter_move(PositionAndIndexIterator<R, T, IndexT> const &it) {
    return (*const_cast<const PositionAndIndexArray<R, T, IndexT> *>(it.array_))[it.offset_];
}

template <size_t R, typename T, typename IndexT>
void swap(PositionAndIndexProxy<R, T, IndexT> lhs, PositionAndIndexProxy<R, T, IndexT> rhs) {
    using std::swap;
    for (size_t i = 0; i < R; ++i) {
        swap(lhs.position[i].get(), rhs.position[i].get());
    }

    swap(lhs.index, rhs.index);
}

} // namespace detail

} // namespace kdtree

} // namespace wenda
