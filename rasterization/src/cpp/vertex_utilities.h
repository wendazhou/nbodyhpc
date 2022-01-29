#pragma once

#include "span.hpp"
#include <array>
#include <vector>

namespace wenda {

struct Vertex {
    float position[3];
    float weight;
    float radius;
};

/** Helper function for rendering data with periodic boundary conditions.
 *
 * This function appends new vertices when the original vertex straddles the boundary,
 * in order to ensure that the periodic boundary conditions appear satisfied when rendering.
 *
 */
void augment_vertices_periodic(std::vector<Vertex> &vertices, std::array<float, 3> const& box_size);

/** Helper function for sorting vertices by z-coordinate.
 *
 * Vertices are required to be sorted prior to rendering.
 *
 */
void sort_vertices(tcb::span<Vertex> vertices);

} // namespace wenda
