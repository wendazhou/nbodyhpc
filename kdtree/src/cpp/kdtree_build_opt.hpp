#pragma once

#include "kdtree.hpp"
#include <immintrin.h>

namespace wenda {

namespace kdtree {

namespace detail {

//! Partitions the given array into elements smaller than the pivot,
//! and elements greater than or equal to the pivot.
//!
//! The index at which the pivot would be placed is returned.
size_t partition_float_array(float *array, size_t n, float pivot);

//! Simple quickselect implementation, uses an AVX2 optimized partitioning subroutine.
void quickselect_float_array(float *array, size_t n, size_t k);

//! Floyd-rivest selection algorithm specialized to float arrays with an optimized AVX2 partitioning
//! subroutine.
void floyd_rivest_float_array(float *array, size_t n, size_t k);

void floyd_rivest_select_loop_position_array(
    PositionAndIndexArray<3> &array, std::ptrdiff_t left, std::ptrdiff_t right, std::ptrdiff_t k,
    int dimension);

void floyd_rivest_select_loop_position_array_avx2(
    PositionAndIndexArray<3> &array, std::ptrdiff_t left, std::ptrdiff_t right, std::ptrdiff_t k,
    int dimension);

struct FloydRivestOptSelectionPolicy {
    typedef PositionAndIndexIterator<3, float, uint32_t> Iter;

    void operator()(Iter beg, Iter med, Iter end, int dimension) const {
        if (med == end) {
            return;
        }

        floyd_rivest_select_loop_position_array(
            *beg.array_, beg.offset_, end.offset_ - 1, med - beg, dimension);
    }
};

struct FloydRivestAvxSelectionPolicy {
    typedef PositionAndIndexIterator<3, float, uint32_t> Iter;

    void operator()(Iter beg, Iter med, Iter end, int dimension) const {
        if (med == end) {
            return;
        }

        floyd_rivest_select_loop_position_array_avx2(
            *beg.array_, beg.offset_, end.offset_ - 1, med - beg, dimension);
    }
};

} // namespace detail

} // namespace kdtree

} // namespace wenda
