/* Modifications Copyright Wenda Zhou 2022.
 *         Copyright Danila Kutenin, 2020-.
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          https://boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>

// Implementation of Floyd-Rivest selection algorithm for fast median computation.

namespace wenda {

namespace kdtree {

namespace detail {

template <
    typename Iter, typename Compare = std::less<>,
    typename DiffType = typename std::iterator_traits<Iter>::difference_type>
inline void floyd_rivest_select_loop(
    Iter begin, DiffType left, DiffType right, DiffType k, Compare const &comp = {}) {

    using std::iter_swap;

    while (right > left) {
        DiffType size = right - left;

        if (size > 600) {
            DiffType n = right - left + 1;
            DiffType i = k - left + 1;

            double z = std::log(n);
            double s = 0.5 * std::exp(2 * z / 3);
            double sd = 0.5 * std::sqrt(z * s * (n - s) / n);
            if (i < n / 2) {
                sd *= -1.0;
            }
            DiffType new_left = std::max(left, static_cast<DiffType>(k - i * s / n + sd));
            DiffType new_right = std::min(right, static_cast<DiffType>(k + (n - i) * s / n + sd));
            floyd_rivest_select_loop<Iter, Compare, DiffType>(begin, new_left, new_right, k, comp);
        }

        DiffType i = left;
        DiffType j = right;

        iter_swap(begin + left, begin + k);

        const bool to_swap = comp(begin[left], begin[right]);
        if (to_swap) {
            iter_swap(begin + left, begin + right);
        }
        // Make sure that non copyable types compile.
        auto t = to_swap ? begin + left : begin + right;
        while (i < j) {
            iter_swap(begin + i, begin + j);
            i++;
            j--;
            while (comp(begin[i], *t)) {
                i++;
            }
            while (comp(*t, begin[j])) {
                j--;
            }
        }

        if (to_swap) {
            iter_swap(begin + left, begin + j);
        } else {
            j++;
            iter_swap(begin + right, begin + j);
        }

        if (j <= k) {
            left = j + 1;
        }
        if (k <= j) {
            right = j - 1;
        }
    }
}

} // namespace detail

template <class Iter, class Compare>
inline void floyd_rivest_select(Iter begin, Iter mid, Iter end, Compare const &comp) {
    if (mid == end)
        return;
    typedef typename std::iterator_traits<Iter>::difference_type DiffType;
    detail::floyd_rivest_select_loop<Iter>(
        begin,
        DiffType{0},
        static_cast<DiffType>(end - begin - 1),
        static_cast<DiffType>(mid - begin),
        comp);
}

} // namespace kdtree

} // namespace wenda
