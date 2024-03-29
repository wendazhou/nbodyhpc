#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    py::array_t<float> const &radii, std::array<float, 3> const& periodic) {
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

    if (periodic[0] > 0 || periodic[1] > 0 || periodic[2] > 0) {
        wenda::augment_vertices_periodic(vertices, periodic);
    }

    wenda::sort_vertices(vertices);

    return vertices;
}

py::array_t<float> render_points(
    wenda::vulkan::PointRenderer &renderer, py::array_t<float> positions, py::array_t<float> weight,
    py::array_t<float> radii, float pixels_per_unit, std::array<float, 3> const& periodic) {

    auto width = renderer.width();
    auto height = renderer.height();

    std::vector<wenda::vulkan::Vertex> vertices = assemble_vertices(positions, weight, radii, periodic);

    float *result_data = new float[width * height];

    {
        py::gil_scoped_release release;
        renderer.render_points(vertices, pixels_per_unit, {result_data, width * height});
    }

    py::capsule free_result(result_data, [](void *ptr) { delete[] static_cast<float *>(ptr); });

    return py::array_t<float>(
        {height, width},
        {sizeof(float), height * sizeof(float)},
        result_data,
        free_result);
}

py::array_t<float> render_points_volume(
    wenda::vulkan::PointRenderer &renderer, py::array_t<float> positions, py::array_t<float> weight,
    py::array_t<float> radii, size_t num_slices, float pixels_per_unit, std::array<float, 3> const& periodic) {

    float *result_data;
    size_t width = renderer.width();
    size_t height = renderer.height();

    std::vector<wenda::vulkan::Vertex> vertices = assemble_vertices(positions, weight, radii, periodic);

    {
        py::gil_scoped_release release;
        result_data = new float[width * height * num_slices];
        renderer.render_points_volume(
            vertices, pixels_per_unit, num_slices,
            {result_data, width * height * num_slices}, check_signals);
    }

    py::capsule free_result(result_data, [](void *ptr) { delete[] static_cast<float *>(ptr); });

    return py::array_t<float>(
        {height, width, num_slices},
        {sizeof(float), height * sizeof(float), width * height * sizeof(float)},
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
                        uint32_t width, uint32_t height,
                        uint32_t subsample_factor) {
                return new wenda::vulkan::PointRenderer(
                    container,
                    {.width = width,
                     .height = height,
                     .subsample_factor = subsample_factor});
            }),
            py::arg("container"),
            py::arg("width"),
            py::arg("height"),
            py::arg("subsample_factor") = 4,
            py::keep_alive<1, 2>{})
        .def_property_readonly("width", &wenda::vulkan::PointRenderer::width)
        .def_property_readonly("height", &wenda::vulkan::PointRenderer::height)
        .def(
            "render_points",
            &render_points,
            py::arg("positions"),
            py::arg("weight"),
            py::arg("radii"),
            py::arg("pixels_per_unit") = 1.0f,
            py::arg("periodic") = std::array<float, 3>{-1, -1, -1})
        .def(
            "render_points_volume",
            &render_points_volume,
            py::arg("positions"),
            py::arg("weight"),
            py::arg("radii"),
            py::arg("num_slices"),
            py::arg("pixels_per_unit") = 1.0f,
            py::arg("periodic") = std::array<float, 3>{-1, -1, -1});
}
