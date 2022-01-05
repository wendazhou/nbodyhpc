#pragma once

#include "span.hpp"

namespace wenda {

namespace kdtree {

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
