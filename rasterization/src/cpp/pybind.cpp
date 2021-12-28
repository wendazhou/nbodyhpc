#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "point_renderer.h"
#include "vertex_utilities.h"
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
    py::array_t<float> const &radii, float box_size, bool periodic) {
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

    if (periodic) {
        wenda::augment_vertices_periodic(vertices, box_size);
    }

    wenda::sort_vertices(vertices);

    return vertices;
}

py::array_t<float> render_points(
    wenda::vulkan::PointRenderer &renderer, py::array_t<float> positions, py::array_t<float> weight,
    py::array_t<float> radii, float box_size, bool periodic) {

    auto grid_size = renderer.grid_size();

    std::vector<wenda::vulkan::Vertex> vertices = assemble_vertices(positions, weight, radii, box_size, periodic);

    float *result_data = new float[grid_size * grid_size];

    {
        py::gil_scoped_release release;
        renderer.render_points(vertices, box_size, {result_data, grid_size * grid_size});
    }

    py::capsule free_result(result_data, [](void *ptr) { delete[] static_cast<float *>(ptr); });

    return py::array_t<float>(
        {(int)grid_size, (int)grid_size},
        {(int)grid_size * sizeof(float), sizeof(float)},
        result_data,
        free_result);
}

py::array_t<float> render_points_volume(
    wenda::vulkan::PointRenderer &renderer, py::array_t<float> positions, py::array_t<float> weight,
    py::array_t<float> radii, float box_size, bool periodic) {
    std::vector<wenda::vulkan::Vertex> vertices = assemble_vertices(positions, weight, radii, box_size, periodic);

    float *result_data;
    long unsigned int grid_size = renderer.grid_size();

    {
        py::gil_scoped_release release;
        result_data = new float[grid_size * grid_size * grid_size];
        renderer.render_points_volume(
            vertices, box_size, {result_data, grid_size * grid_size * grid_size}, check_signals);
    }

    py::capsule free_result(result_data, [](void *ptr) { delete[] static_cast<float *>(ptr); });

    return py::array_t<float>(
        {grid_size, grid_size, grid_size},
        {grid_size * grid_size * sizeof(float), grid_size * sizeof(float), sizeof(float)},
        result_data,
        free_result);
}

} // namespace

PYBIND11_MODULE(_impl, m) {
    m.doc() = "Vulkan-based rasterization for fast field generation";

    py::class_<wenda::vulkan::VulkanContainer>(m, "VulkanContainer")
        .def(py::init<bool>(), py::arg("enable_validation_layers") = false);

    py::class_<wenda::vulkan::PointRenderer>(m, "PointRenderer")
        .def(
            py::init([](wenda::vulkan::VulkanContainer const &container,
                        uint32_t grid_size,
                        uint32_t subsample_factor) {
                return new wenda::vulkan::PointRenderer(
                    container,
                    {.grid_size = grid_size,
                     .subsample_factor = subsample_factor});
            }),
            py::arg("container"),
            py::arg("grid_size"),
            py::arg("subsample_factor") = 4)
        .def_property_readonly("grid_size", &wenda::vulkan::PointRenderer::grid_size)
        .def(
            "render_points",
            &render_points,
            py::arg("positions"),
            py::arg("weight"),
            py::arg("radii"),
            py::arg("box_size") = 1.0f,
            py::arg("periodic") = false)
        .def(
            "render_points_volume",
            &render_points_volume,
            py::arg("positions"),
            py::arg("weight"),
            py::arg("radii"),
            py::arg("box_size") = 1.0f,
            py::arg("periodic") = false);
}
