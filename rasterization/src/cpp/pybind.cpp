#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "point_renderer.h"
#include "vulkan_support.h"

namespace py = pybind11;

namespace {

bool check_signals() {
    py::gil_scoped_acquire acquire;

    if (PyErr_CheckSignals() != 0) {
        throw py::error_already_set();
    }

    return false;
}

std::vector<wenda::vulkan::Vertex> assemble_vertices(
    py::array_t<float> const &positions, py::array_t<float> const &weights,
    py::array_t<float> const &radii) {
    if (positions.ndim() != 2 || positions.shape(1) != 3) {
        throw std::runtime_error("positions must be a 2D array of shape (N, 3)");
    }

    if (weights.ndim() != 1) {
        throw std::runtime_error("weight must be a 1D array");
    }

    if (radii.ndim() != 1) {
        throw std::runtime_error("radii must be a 1D array");
    }

    if (radii.shape(0) != positions.shape(0)) {
        throw std::runtime_error("radii must have the same length as positions");
    }

    if (weights.shape(0) != positions.shape(0)) {
        throw std::runtime_error("weights must have the same length as positions");
    }

    auto positions_view = positions.unchecked<2>();
    auto weights_view = weights.unchecked<1>();
    auto radii_view = radii.unchecked<1>();

    auto num_points = positions.shape(0);

    std::vector<wenda::vulkan::Vertex> vertices(num_points);

    for (auto i = 0; i < num_points; ++i) {
        vertices[i].position[0] = positions_view(i, 0);
        vertices[i].position[1] = positions_view(i, 1);
        vertices[i].position[2] = positions_view(i, 2);
        vertices[i].weight = weights_view(i);
        vertices[i].radius = radii_view(i);
    }

    std::sort(vertices.begin(), vertices.end(),
              [](wenda::vulkan::Vertex const &a, wenda::vulkan::Vertex const &b) {
                  return a.position[2] < b.position[2];
              });

    return vertices;
}

py::array_t<float> render_points(
    py::array_t<float> positions, py::array_t<float> weight, py::array_t<float> radii,
    float box_size, uint32_t grid_size) {
    std::vector<wenda::vulkan::Vertex> vertices = assemble_vertices(positions, weight, radii);

    wenda::vulkan::VulkanContainer vulkan;
    wenda::vulkan::PointRenderer renderer(vulkan, box_size, grid_size);

    float *result_data = new float[grid_size * grid_size];
    renderer.render_points(vertices, {result_data, grid_size * grid_size});

    py::capsule free_result(result_data, [](void *ptr) { delete[] static_cast<float *>(ptr); });

    return py::array_t<float>(
        {(int)grid_size, (int)grid_size},
        {(int)grid_size * sizeof(float), sizeof(float)},
        result_data,
        free_result);
}

py::array_t<float> render_points_volume(
    py::array_t<float> positions, py::array_t<float> weight, py::array_t<float> radii,
    float box_size, uint32_t grid_size) {
    std::vector<wenda::vulkan::Vertex> vertices = assemble_vertices(positions, weight, radii);

    float *result_data;

    {
        py::gil_scoped_release release;

        wenda::vulkan::VulkanContainer vulkan;
        wenda::vulkan::PointRenderer renderer(vulkan, box_size, grid_size);

        result_data = new float[grid_size * grid_size * grid_size];
        renderer.render_points_volume(
            vertices, {result_data, grid_size * grid_size * grid_size}, check_signals);
    }

    py::capsule free_result(result_data, [](void *ptr) { delete[] static_cast<float *>(ptr); });

    long int grid_size_i = (int)grid_size;

    return py::array_t<float>(
        {grid_size_i, grid_size_i, grid_size_i},
        {grid_size_i * grid_size_i * sizeof(float), grid_size_i * sizeof(float), sizeof(float)},
        result_data,
        free_result);
}

} // namespace

PYBIND11_MODULE(_impl, m) {
    m.doc() = "Vulkan-based rasterization for fast field generation";

    m.def(
        "render_points",
        &render_points,
        "Render points to image",
        py::arg("positions"),
        py::arg("weight"),
        py::arg("radii"),
        py::arg("box_size"),
        py::arg("grid_size"));

    m.def(
        "render_points_volume",
        &render_points_volume,
        "Render points to volume",
        py::arg("positions"),
        py::arg("weight"),
        py::arg("radii"),
        py::arg("box_size"),
        py::arg("grid_size"));
}
