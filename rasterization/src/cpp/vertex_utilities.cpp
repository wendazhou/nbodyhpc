#include "vertex_utilities.h"

#include <algorithm>

namespace wenda {

void sort_vertices(tcb::span<Vertex> vertices) {
    std::sort(vertices.begin(), vertices.end(), [](Vertex const &a, Vertex const &b) {
        return a.position[2] < b.position[2];
    });
}

void augment_vertices_periodic(std::vector<Vertex> &vertices, float box_size) {
    auto num_vertices = vertices.size();

    for (size_t i = 0; i < num_vertices; ++i) {
        auto vertex = vertices[i];
        auto radius = vertex.radius;

        for (int dim = 0; dim < 3; ++dim) {
            auto pos_dim = vertex.position[dim];

            if (pos_dim + radius > box_size) {
                auto new_vertex = vertex;
                new_vertex.position[dim] = pos_dim - box_size;
                vertices.push_back(new_vertex);
            }

            if (pos_dim - radius < 0.0f) {
                auto new_vertex = vertex;
                new_vertex.position[dim] = pos_dim + box_size;
                vertices.push_back(new_vertex);
            }
        }
    }
}
} // namespace wenda
