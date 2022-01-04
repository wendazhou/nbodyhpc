#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_impl, m) {
    m.doc() = "Fast KD-tree for spatial data, including periodic boundary conditions.";
}
