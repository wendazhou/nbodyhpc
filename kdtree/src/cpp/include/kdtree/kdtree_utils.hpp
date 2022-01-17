#pragma once

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include "kdtree.hpp"

namespace wenda {
namespace kdtree {

template <size_t R = 3>
inline std::vector<PositionAndIndex<R>>
make_random_position_and_index(uint32_t n, unsigned int seed, float boxsize = 1.0) {
    std::vector<PositionAndIndex<R>> positions(n);

    typedef r123::Philox4x32 RNG;
    RNG rng;

    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = seed;

    for (uint32_t i = 0; i < n; ++i) {
        c.v[1] = i;
        auto r = rng(c, uk);

        for (size_t dim = 0; dim < R; ++dim) {
            c.v[0] = dim;
            auto r = rng(c, uk);
            positions[i].position[dim] = r123::u01<float>(r[0]) * boxsize;
        }

        positions[i].index = i;
    }

    return positions;
}

/** Generates a PositionAndIndexArray with random positions.
 *
 * @param n Number of elements in the array.
 * @param seed Seed for the random number generator.
 * @param boxsize Size of the box in each dimension.
 * @param block_size If positive, pads the array to a multiple of the given block size.
 *
 */
template <size_t R = 3, typename T = float, typename IndexT = uint32_t>
inline PositionAndIndexArray<R, T, IndexT> make_random_position_and_index_array(
    uint32_t n, unsigned int seed, float boxsize = 1.0, int block_size = -1) {

    uint32_t size_up;
    if (block_size <= 0) {
        size_up = n;
    } else {
        size_up = (n + block_size - 1) / block_size * block_size;
    }

    PositionAndIndexArray<R, T, IndexT> result(size_up);

    typedef r123::Philox4x32 RNG;
    RNG rng;

    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = seed;

    for (size_t dim = 0; dim < R; ++dim) {
        c.v[0] = dim;

        for (uint32_t i = 0; i < n; ++i) {
            c.v[1] = i;
            auto r = rng(c, uk);

            result.positions_[dim][i] = r123::u01<T>(r[0]) * boxsize;
        }

        for (uint32_t i = n; i < size_up; ++i) {
            result.positions_[dim][i] = std::numeric_limits<T>::max();
        }
    }

    std::iota(result.indices_.begin(), result.indices_.end(), 0);

    return result;
}

/** Utility function to access underlying container of a STL container adapter.
 * This function is used to get around the access protections of the STL container adapters,
 * and expose the protected member c.
 *
 * @param adapter The STL container adapter to access.
 *
 */
template <class ContainerAdapter>
typename ContainerAdapter::container_type &get_container_from_adapter(ContainerAdapter &adapter) {
    struct hack : ContainerAdapter {
        static typename ContainerAdapter::container_type &get(ContainerAdapter &a) {
            return a.*&hack::c;
        }
    };

    return hack::get(adapter);
}

/** Utility function to make an array of position and indices from the given positions.
 *
 * @param positions The positions to convert.
 * @param block_size If positive, pads the array to a multiple of the given block size.
 *
 */
PositionAndIndexArray<3, float, uint32_t> make_position_and_indices(
    tcb::span<const std::array<float, 3>> const &positions, int block_size = -1);

} // namespace kdtree
} // namespace wenda
