#pragma once

#include "kdtree.hpp"

namespace wenda {

namespace kdtree {

namespace detail {

void floyd_rivest_select_loop_position_array(
    PositionAndIndexArray<3>& array, std::ptrdiff_t left, std::ptrdiff_t right,
    std::ptrdiff_t k, int dimension) {
    using std::iter_swap;

    auto& positions = array.positions_;

    auto comp = [&,dimension](ptrdiff_t left, ptrdiff_t right) {
        return positions[dimension][left] < positions[dimension][right];
    };

    while (right > left) {
        std::ptrdiff_t size = right - left;

        if (size > 600) {
            std::ptrdiff_t n = right - left + 1;
            std::ptrdiff_t i = k - left + 1;

            double z = std::log(n);
            double s = 0.5 * std::exp(2 * z / 3);
            double sd = 0.5 * std::sqrt(z * s * (n - s) / n);
            if (i < n / 2) {
                sd *= -1.0;
            }
            std::ptrdiff_t new_left = std::max(left, static_cast<std::ptrdiff_t>(k - i * s / n + sd));
            std::ptrdiff_t new_right = std::min(right, static_cast<std::ptrdiff_t>(k + (n - i) * s / n + sd));
            floyd_rivest_select_loop_position_array(array, new_left, new_right, k, dimension);
        }

        std::ptrdiff_t i = left;
        std::ptrdiff_t j = right;

        array.swap_elements(left, k);

        const bool to_swap = comp(left, right);
        if (to_swap) {
            array.swap_elements(left, right);
        }

        auto t = to_swap ? left : right;
        while (i < j) {
            array.swap_elements(i, j);
            i++;
            j--;
            while (comp(i, t)) {
                i++;
            }
            while (comp(t, j)) {
                j--;
            }
        }

        if (to_swap) {
            array.swap_elements(left, j);
        } else {
            j++;
            array.swap_elements(right, j);
        }

        if (j <= k) {
            left = j + 1;
        }
        if (k <= j) {
            right = j - 1;
        }
    }
}

struct FloydRivestOptSelectionPolicy {
    typedef PositionAndIndexIterator<3, float, uint32_t> Iter;

    void operator()(Iter beg, Iter med, Iter end, int dimension) const {
        if(med == end) {
            return;
        }

        floyd_rivest_select_loop_position_array(*beg.array_, beg.offset_, end - beg - 1, med - beg, dimension);
    }
};

} // namespace detail

} // namespace kdtree

} // namespace wenda
