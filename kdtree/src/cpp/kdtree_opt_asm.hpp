#pragma once

#include "kdtree.hpp"
#include "kdtree_opt.hpp"
#include "tournament_tree.hpp"

extern "C" {
void wenda_insert_closest_l2_avx2(void const *positions, size_t n, float const *query, void *tree);
}

namespace wenda {

namespace kdtree {

namespace {

template <typename DistanceT, typename QueueT>
struct InsertShorterDistanceAsmAvx2 : InsertShorterDistanceVanilla<DistanceT, QueueT> {};

#ifdef _WIN32
// Only implemented on Windows at the moment

template <>
struct InsertShorterDistanceAsmAvx2<
    L2Distance, TournamentTree<std::pair<float, uint32_t>, PairLessFirst>> {
    typedef L2Distance distance_t;
    typedef std::pair<float, uint32_t> result_t;
    typedef TournamentTree<std::pair<float, uint32_t>, PairLessFirst> queue_t;

    void operator()(
        tcb::span<const PositionAndIndex> positions, std::array<float, 3> const &query,
        queue_t &distances, L2Distance const &) const {

        wenda_insert_closest_l2_avx2(
            positions.data(), positions.size(), query.data(), distances.data().data());
    }
};

#endif

} // namespace

} // namespace kdtree

} // namespace wenda
