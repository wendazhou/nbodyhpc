#include "vertex_utilities.h"

#include <algorithm>

namespace wenda {

void sort_vertices(tcb::span<Vertex> vertices) {
    std::sort(vertices.begin(), vertices.end(), [](Vertex const &a, Vertex const &b) {
        return a.position[2] < b.position[2];
    });
}

void augment_vertices_periodic(std::vector<Vertex> &vertices, std::array<float, 3> const& box_size) {
    for (int dim = 0; dim < 3; ++dim) {
        auto num_vertices = vertices.size();
        auto box_size_dim = box_size[dim];

        if (box_size_dim <= 0) {
            // Negative box size indicates that the periodic boundary condition
            // is not active for this dimension
            continue;
        }

        for (size_t i = 0; i < num_vertices; ++i) {
            auto vertex = vertices[i];
            auto radius = vertex.radius;

            auto pos_dim = vertex.position[dim];

            if (pos_dim + radius > box_size_dim) {
                auto new_vertex = vertex;
                new_vertex.position[dim] = pos_dim - box_size_dim;
                vertices.push_back(new_vertex);
            }

            if (pos_dim - radius < 0.0f) {
                auto new_vertex = vertex;
                new_vertex.position[dim] = pos_dim + box_size_dim;
                vertices.push_back(new_vertex);
            }
        }
    }
}
} // namespace wenda
