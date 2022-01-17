#pragma once

#include <cassert>
#include <cstdint>

#include "kdtree.hpp"
#include "kdtree_opt.hpp"
#include "tournament_tree.hpp"

extern "C" {
void wenda_insert_closest_l2_avx2(
    float const *const *positions, size_t n, float const *query, void *tree,
    uint32_t const *indices);
void wenda_insert_closest_l2_periodic_avx2(
    float const *const *positions, size_t n, float const *query, void *tree,
    uint32_t const *indices, float boxsize);
}

namespace wenda {

namespace kdtree {

template <typename DistanceT, typename QueueT> struct InsertShorterDistanceAsm;

template <>
struct InsertShorterDistanceAsm<
    L2Distance, TournamentTree<std::pair<float, uint32_t>, PairLessFirst>> {
    typedef L2Distance distance_t;
    typedef TournamentTree<std::pair<float, uint32_t>, PairLessFirst> queue_t;
    typedef std::pair<float, uint32_t> result_t;

    void operator()(
        OffsetRangeContainerWrapper<const PositionAndIndexArray<3>> const &positions,
        std::array<float, 3> const &query, queue_t &distances,
        distance_t const &distance_func) const {

        assert(positions.size() % 8 == 0);

        std::array<float const *, 3> positions_offset;
        std::transform(
            positions.container_.positions_.begin(),
            positions.container_.positions_.end(),
            positions_offset.begin(),
            [&](auto const &p) { return p + positions.offset; });

        assert(reinterpret_cast<std::ptrdiff_t>(positions_offset[0]) % 32 == 0);
        assert(reinterpret_cast<std::ptrdiff_t>(positions_offset[1]) % 32 == 0);
        assert(reinterpret_cast<std::ptrdiff_t>(positions_offset[2]) % 32 == 0);

        wenda_insert_closest_l2_avx2(
            positions_offset.data(),
            positions.size(),
            query.data(),
            distances.data().data(),
            positions.container_.indices_.data() + positions.offset);
    }
};


template <>
struct InsertShorterDistanceAsm<
    L2PeriodicDistance<float>, TournamentTree<std::pair<float, uint32_t>, PairLessFirst>> {
    typedef L2PeriodicDistance<float> distance_t;
    typedef TournamentTree<std::pair<float, uint32_t>, PairLessFirst> queue_t;
    typedef std::pair<float, uint32_t> result_t;

    void operator()(
        OffsetRangeContainerWrapper<const PositionAndIndexArray<3>> const &positions,
        std::array<float, 3> const &query, queue_t &distances,
        distance_t const &distance_func) const {

        assert(positions.size() % 8 == 0);

        std::array<float const *, 3> positions_offset;
        std::transform(
            positions.container_.positions_.begin(),
            positions.container_.positions_.end(),
            positions_offset.begin(),
            [&](auto const &p) { return p + positions.offset; });

        assert(reinterpret_cast<std::ptrdiff_t>(positions_offset[0]) % 32 == 0);
        assert(reinterpret_cast<std::ptrdiff_t>(positions_offset[1]) % 32 == 0);
        assert(reinterpret_cast<std::ptrdiff_t>(positions_offset[2]) % 32 == 0);

        wenda_insert_closest_l2_periodic_avx2(
            positions_offset.data(),
            positions.size(),
            query.data(),
            distances.data().data(),
            positions.container_.indices_.data() + positions.offset,
            distance_func.box_size_);
    }
};

} // namespace kdtree

} // namespace wenda
