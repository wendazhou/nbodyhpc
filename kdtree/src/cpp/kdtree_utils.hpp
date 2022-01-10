#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include "kdtree.hpp"

namespace wenda {
namespace kdtree {

inline std::vector<PositionAndIndex> make_random_position_and_index(uint32_t n, unsigned int seed, float boxsize=1.0) {
    std::vector<PositionAndIndex> positions(n);

    typedef r123::Philox4x32 RNG;
    RNG rng;

    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = seed;

    for (uint32_t i = 0; i < n; ++i) {
        c.v[0] = i;
        auto r = rng(c, uk);

        positions[i].position[0] = r123::u01<float>(r[0]) * boxsize;
        positions[i].position[1] = r123::u01<float>(r[1]) * boxsize;
        positions[i].position[2] = r123::u01<float>(r[2]) * boxsize;
        positions[i].index = i;
    }

    return positions;
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
 * 
 */
std::vector<PositionAndIndex> make_position_and_indices(tcb::span<const std::array<float, 3>> const &positions);

} // namespace kdtree
} // namespace wenda
