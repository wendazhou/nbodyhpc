#include <optional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kdtree.hpp"
#include <thread_pool.hpp>

namespace py = pybind11;

namespace {

wenda::kdtree::PositionAndIndexArray<3, float, uint32_t> make_positions_and_indices(
    py::array_t<float> const &points, std::optional<float> box_size, int block_size) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::runtime_error("positions must be a 2D array of shape (N, 3)");
    }

    auto points_view = points.unchecked<2>();
    auto num_points = points.shape(0);

    auto size_up = (num_points + block_size - 1) / block_size * block_size;

    wenda::kdtree::PositionAndIndexArray<3, float, uint32_t> positions(size_up);

    std::iota(positions.indices_.begin(), positions.indices_.end(), 0);

    // Pad to next multiple of block_size
    for (size_t dim = 0; dim < 3; ++dim) {
        for (size_t i = 0; i < num_points; ++i) {
            positions.positions_[dim][i] = points_view(i, dim);
        }

        if (box_size) {
            auto box_size_value = *box_size;
            auto all_in_box = std::all_of(
                positions.positions_[dim], positions.positions_[dim] + num_points, [&](float x) {
                    return x >= 0.0f && x <= box_size_value;
                });

            if (!all_in_box) {
                throw std::runtime_error(
                    "When using periodic boundary conditions, all points must be "
                    "within the box (0 <= x <= box_size).");
            }
        }

        std::fill(
            positions.positions_[dim] + num_points,
            positions.positions_[dim] + size_up,
            std::numeric_limits<float>::max());
    }

    return positions;
}

} // namespace

namespace wenda {
namespace kdtree {

//! KDTree wrapper for pybind compatibility
class PyKDTree : public wenda::kdtree::KDTree {
    bool periodic_;
    float box_size_;

  public:
    PyKDTree(PyKDTree &&) noexcept = default;

    size_t num_points() const noexcept { return positions().size(); }
    size_t num_nodes() const noexcept { return nodes().size(); }
    bool periodic() const noexcept { return periodic_; }
    float box_size() const noexcept { return box_size_; }

    PyKDTree(
        py::array_t<float> points, int leaf_size, int max_threads, std::optional<float> box_size)
        : KDTree(
              make_positions_and_indices(points, box_size, 8),
              {.leaf_size = leaf_size, .max_threads = max_threads, .block_size = 8}),
          periodic_(box_size.has_value()), box_size_(box_size.value_or(0.0f)) {}

    static PyKDTree from_points(
        py::array_t<float> const &points, int leaf_size, int max_threads,
        std::optional<float> box_size) {
        py::gil_scoped_release nogil;
        return PyKDTree(points, leaf_size, max_threads, box_size);
    }

    std::pair<py::array_t<float>, py::array_t<uint32_t>>
    query(py::array_t<float> points, int k, int workers) {
        if (k <= 0) {
            throw std::runtime_error("k must be positive integer");
        }

        if (points.ndim() != 2 || points.shape(1) != 3) {
            throw std::runtime_error("positions must be a 2D array of shape (N, 3)");
        }

        auto num_points = points.shape(0);
        auto points_view = points.unchecked<2>();

        float *result_distances = new float[num_points * k];
        uint32_t *result_indices = new uint32_t[num_points * k];

        std::function<void(size_t, size_t)> loop_fn;

        if (periodic_) {
            auto distance_periodic = wenda::kdtree::L2PeriodicDistance<float>{box_size_};

            loop_fn = [&, distance_periodic](size_t start, size_t end) {
                for (size_t i = start; i < end; ++i) {
                    auto result = find_closest(
                        {points_view(i, 0), points_view(i, 1), points_view(i, 2)},
                        k,
                        distance_periodic);

                    std::transform(
                        result.begin(), result.end(), result_distances + i * k, [](auto const &r) {
                            return r.first;
                        });
                    std::transform(
                        result.begin(), result.end(), result_indices + i * k, [](auto const &r) {
                            return r.second;
                        });

                    // check for signals set by python
                    if (i % 1000 == 0) {
                        py::gil_scoped_acquire gil;
                        if (PyErr_CheckSignals() != 0) {
                            throw py::error_already_set();
                        }
                    }
                }
            };
        } else {
            loop_fn = [&](size_t start, size_t end) {
                for (size_t i = start; i < end; ++i) {
                    auto result = find_closest(
                        {points_view(i, 0), points_view(i, 1), points_view(i, 2)},
                        k,
                        wenda::kdtree::L2Distance{});

                    std::transform(
                        result.begin(), result.end(), result_distances + i * k, [](auto const &r) {
                            return r.first;
                        });
                    std::transform(
                        result.begin(), result.end(), result_indices + i * k, [](auto const &r) {
                            return r.second;
                        });

                    // check for signals set by python
                    if (i % 1000 == 0) {
                        py::gil_scoped_acquire gil;
                        if (PyErr_CheckSignals() != 0) {
                            throw py::error_already_set();
                        }
                    }
                }
            };
        }

        if (workers == 1) {
            // workers = 1 indicates that no threading is desired.
            py::gil_scoped_release nogil;
            loop_fn(0, num_points);
        } else {
            wenda::thread_pool pool(workers > 0 ? workers : std::thread::hardware_concurrency());
            py::gil_scoped_release nogil;
            pool.parallelize_loop(0, num_points, loop_fn);
        }

        py::capsule free_result_distances(
            result_distances, [](void *ptr) { delete[] static_cast<float *>(ptr); });
        py::capsule free_result_indices(
            result_indices, [](void *ptr) { delete[] static_cast<uint32_t *>(ptr); });

        py::array::ShapeContainer shape{(py::ssize_t)num_points, (py::ssize_t)k};

        return {
            py::array_t<float>{
                shape, {k * sizeof(float), sizeof(float)}, result_distances, free_result_distances},
            py::array_t<uint32_t>{
                shape,
                {k * sizeof(uint32_t), sizeof(uint32_t)},
                result_indices,
                free_result_indices}};
    }
};

} // namespace kdtree

} // namespace wenda

PYBIND11_MODULE(_impl, m) {
    m.doc() = "Fast KD-tree for spatial data, including periodic boundary conditions.";

    py::class_<wenda::kdtree::PyKDTree>(m, "KDTree")
        .def(
            py::init(&wenda::kdtree::PyKDTree::from_points),
            py::arg("points"),
            py::arg("leafsize") = 64,
            py::arg("max_threads") = -1,
            py::arg("boxsize") = std::nullopt)
        .def(
            "query",
            &wenda::kdtree::PyKDTree::query,
            py::arg("points"),
            py::arg("k") = 1,
            py::arg("workers") = 1)
        .def_property_readonly("n", &wenda::kdtree::PyKDTree::num_points)
        .def_property_readonly("size", &wenda::kdtree::PyKDTree::num_nodes)
        .def_property_readonly("periodic", &wenda::kdtree::PyKDTree::periodic)
        .def_property_readonly("boxsize", &wenda::kdtree::PyKDTree::box_size);
}
