#include "kdtree_build_opt.hpp"

#include <cmath>

#include <Random123/philox.h>

namespace {

extern const __m256i partition_permutation_masks[256];

inline float calc_min(__m256 vec) { /* minimum of 8 floats */
    auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    vec = _mm256_min_ps(vec, _mm256_permutevar8x32_ps(vec, perm_mask));
    vec = _mm256_min_ps(vec, _mm256_shuffle_ps(vec, vec, 0b10110001));
    vec = _mm256_min_ps(vec, _mm256_shuffle_ps(vec, vec, 0b01001110));
    return _mm_extract_ps(_mm256_castps256_ps128(vec), 0);
}

inline float calc_max(__m256 vec) { /* maximum of 8 floats */
    auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    vec = _mm256_max_ps(vec, _mm256_permutevar8x32_ps(vec, perm_mask));
    vec = _mm256_max_ps(vec, _mm256_shuffle_ps(vec, vec, 0b10110001));
    vec = _mm256_max_ps(vec, _mm256_shuffle_ps(vec, vec, 0b01001110));
    return _mm_extract_ps(_mm256_castps256_ps128(vec), 0);
}

struct DummyData {
    void permute(__m256i perm) {}
    DummyData load(ptrdiff_t idx) const { return DummyData{}; }
    void store(DummyData const &, ptrdiff_t idx) {}
    void swap_elements(ptrdiff_t, ptrdiff_t) {}
};

template <size_t R, typename T, typename IndexT> struct PositionAndIndexData {
    struct Vec {
        __m256 positions[R];
        __m256i indices;

        void permute(__m256i perm) {
            for (size_t i = 0; i < R - 1; ++i) {
                positions[i] = _mm256_permutevar8x32_ps(positions[i], perm);
            }

            indices = _mm256_permutevar8x32_epi32(indices, perm);
        }
    };

    int dimension_;
    std::array<T *, R> positions_;
    IndexT *indices_;

    Vec load(ptrdiff_t idx) const {
        Vec result;

        for (size_t i = 0; i < R; ++i) {
            result.positions[i] = _mm256_loadu_ps(positions_[i] + idx);
        }

        result.indices = _mm256_loadu_si256(reinterpret_cast<__m256i *>(indices_ + idx));
        return result;
    }

    void store(Vec const &d, ptrdiff_t idx) {
        for (size_t i = 0; i < R; ++i) {
            _mm256_storeu_ps(positions_[i] + idx, d.positions[i]);
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i *>(indices_ + idx), d.indices);
    }

    void swap_elements(ptrdiff_t i, ptrdiff_t j) {
        for (size_t k = 0; k < R; ++k) {
            std::swap(positions_[k][i], positions_[k][j]);
        }

        std::swap(indices_[i], indices_[j]);
    }
};

template <typename T = DummyData>
int partition_vec(__m256 &curr_vec, const __m256 &pivot_vec, T &data) {
    __m256 compared = _mm256_cmp_ps(curr_vec, pivot_vec, _CMP_NLT_US);
    int mm = _mm256_movemask_ps(compared);

    auto perm = partition_permutation_masks[mm];

    /* how many ones, each 1 stands for an element greater than pivot */
    int amount_gt_pivot = _mm_popcnt_u32(mm);
    data.permute(perm);

    curr_vec = _mm256_permutevar8x32_ps(curr_vec, perm);

    return amount_gt_pivot;
}

//! Partitions a vector of 8 elements according to the given value.
//! This function uses a pre-computed lookup table.
int partition_vec(__m256 &curr_vec, const __m256 &pivot_vec) {
    DummyData data;
    return partition_vec(curr_vec, pivot_vec, data);
}

int partition_vec(
    __m256 &curr_vec, const __m256 &pivot_vec, __m256 &smallest_vec, __m256 &biggest_vec) {
    /* update the smallest and largest values of the array */
    smallest_vec = _mm256_min_ps(curr_vec, smallest_vec);
    biggest_vec = _mm256_max_ps(curr_vec, biggest_vec);

    return partition_vec(curr_vec, pivot_vec);
}

template <typename T = DummyData>
inline size_t
partition_vectorized_8(float *arr, ptrdiff_t left, ptrdiff_t right, float pivot, T &data) {
    /* make array length divisible by eight, shortening the array */
    for (auto i = (right - left) % 8; i > 0; --i) {
        if (arr[left] < pivot) {
            ++left;
        } else {
            std::swap(arr[left], arr[--right]);
            data.swap_elements(left, right);
        }
    }

    if (left == right)
        return left; /* less than 8 elements in the array */

    auto pivot_vec = _mm256_set1_ps(pivot); /* fill vector with pivot */

    if (right - left == 8) { /* if 8 elements left after shortening */
        auto v = _mm256_loadu_ps(arr + left);
        auto d = data.load(left);

        int amount_gt_pivot = partition_vec(v, pivot_vec, d);

        _mm256_storeu_ps(arr + left, v);
        data.store(d, left);

        return left + (8 - amount_gt_pivot);
    }

    /* first and last 8 values are partitioned at the end */
    auto vec_left = _mm256_loadu_ps(arr + left);         /* first 8 values */
    auto vec_right = _mm256_loadu_ps(arr + (right - 8)); /* last 8 values  */

    auto d_left = data.load(left);
    auto d_right = data.load(right - 8);

    /* store points of the vectors */
    int r_store = right - 8; /* right store point */
    int l_store = left;      /* left store point */
    /* indices for loading the elements */
    left += 8;  /* increase, because first 8 elements are cached */
    right -= 8; /* decrease, because last 8 elements are cached */

    while (right - left != 0) {  /* partition 8 elements per iteration */
        __m256 curr_vec;         /* vector to be partitioned */
        decltype(d_left) d_curr; /* data to be partitioned */

        /* if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side */
        if ((r_store + 8) - right < left - l_store) {
            right -= 8;
            curr_vec = _mm256_loadu_ps(arr + right);
            d_curr = data.load(right);
        } else {
            curr_vec = _mm256_loadu_ps(arr + left);
            d_curr = data.load(left);
            left += 8;
        }
        /* partition the current vector and save it on both sides of the array */
        int amount_gt_pivot = partition_vec(curr_vec, pivot_vec, d_curr);

        _mm256_storeu_ps(arr + l_store, curr_vec);
        _mm256_storeu_ps(arr + r_store, curr_vec);

        data.store(d_curr, l_store);
        data.store(d_curr, r_store);

        /* update store points */
        r_store -= amount_gt_pivot;
        l_store += (8 - amount_gt_pivot);
    }

    /* partition and save vec_left */
    int amount_gt_pivot = partition_vec(vec_left, pivot_vec, d_left);
    _mm256_storeu_ps(arr + l_store, vec_left);
    _mm256_storeu_ps(arr + r_store, vec_left);
    data.store(d_left, l_store);
    data.store(d_left, r_store);

    l_store += (8 - amount_gt_pivot);
    /* partition and save vec_right */
    amount_gt_pivot = partition_vec(vec_right, pivot_vec, d_right);
    _mm256_storeu_ps(arr + l_store, vec_right);
    data.store(d_right, l_store);
    l_store += (8 - amount_gt_pivot);

    return l_store;
}

inline size_t partition_vectorized_8(
    float *arr, ptrdiff_t left, ptrdiff_t right, float pivot, float &smallest, float &biggest) {
    /* make array length divisible by eight, shortening the array */
    for (auto i = (right - left) % 8; i > 0; --i) {
        smallest = std::min(smallest, arr[left]);
        biggest = std::max(biggest, arr[left]);
        if (arr[left] < pivot) {
            ++left;
        } else {
            std::swap(arr[left], arr[--right]);
        }
    }

    if (left == right)
        return left; /* less than 8 elements in the array */

    auto pivot_vec = _mm256_set1_ps(pivot); /* fill vector with pivot */
    auto sv = _mm256_set1_ps(smallest);     /* vector for smallest elements */
    auto bv = _mm256_set1_ps(biggest);      /* vector for biggest elements */

    if (right - left == 8) { /* if 8 elements left after shortening */
        auto v = _mm256_loadu_ps(arr + left);
        int amount_gt_pivot = partition_vec(v, pivot_vec, sv, bv);
        _mm256_storeu_ps(arr + left, v);
        smallest = calc_min(sv);
        biggest = calc_max(bv);
        return left + (8 - amount_gt_pivot);
    }

    /* first and last 8 values are partitioned at the end */
    auto vec_left = _mm256_loadu_ps(arr + left);         /* first 8 values */
    auto vec_right = _mm256_loadu_ps(arr + (right - 8)); /* last 8 values  */
    /* store points of the vectors */
    int r_store = right - 8; /* right store point */
    int l_store = left;      /* left store point */
    /* indices for loading the elements */
    left += 8;  /* increase, because first 8 elements are cached */
    right -= 8; /* decrease, because last 8 elements are cached */

    while (right - left != 0) { /* partition 8 elements per iteration */
        __m256 curr_vec;        /* vector to be partitioned */
        /* if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side */
        if ((r_store + 8) - right < left - l_store) {
            right -= 8;
            curr_vec = _mm256_loadu_ps(arr + right);
        } else {
            curr_vec = _mm256_loadu_ps(arr + left);
            left += 8;
        }
        /* partition the current vector and save it on both sides of the array */
        int amount_gt_pivot = partition_vec(curr_vec, pivot_vec, sv, bv);
        ;
        _mm256_storeu_ps(arr + l_store, curr_vec);
        _mm256_storeu_ps(arr + r_store, curr_vec);
        /* update store points */
        r_store -= amount_gt_pivot;
        l_store += (8 - amount_gt_pivot);
    }

    /* partition and save vec_left */
    int amount_gt_pivot = partition_vec(vec_left, pivot_vec, sv, bv);
    _mm256_storeu_ps(arr + l_store, vec_left);
    _mm256_storeu_ps(arr + r_store, vec_left);
    l_store += (8 - amount_gt_pivot);
    /* partition and save vec_right */
    amount_gt_pivot = partition_vec(vec_right, pivot_vec, sv, bv);
    _mm256_storeu_ps(arr + l_store, vec_right);
    l_store += (8 - amount_gt_pivot);

    smallest = calc_min(sv); /* determine smallest value in vector */
    biggest = calc_max(bv);  /* determine largest value in vector */
    return l_store;
}

void floyd_rivest_select_float_loop(float *array, ptrdiff_t left, ptrdiff_t right, ptrdiff_t k) {
    using std::iter_swap;

    while (right > left) {
        auto size = right - left;

        if (size > 600) {
            auto n = right - left + 1;
            auto i = k - left + 1;

            double z = std::log(n);
            double s = 0.5 * std::exp(2 * z / 3);
            double sd = 0.5 * std::sqrt(z * s * (n - s) / n);

            // We alternate choosing left and right pivots
            if (i < n / 2) {
                sd *= -1.0;
            }
            auto new_left = std::max(left, static_cast<ptrdiff_t>(k - i * s / n + sd));
            auto new_right = std::min(right, static_cast<ptrdiff_t>(k + (n - i) * s / n + sd));
            floyd_rivest_select_float_loop(array, new_left, new_right, k);
        }

        // Load the chosen pivot element
        auto t = array[k];

        // Place chosen pivot at end
        iter_swap(array + right, array + k);

        ptrdiff_t pivot_index =
            wenda::kdtree::detail::partition_float_array(array + left, right - left, t) + left;

        iter_swap(array + pivot_index, array + right);

        // Recurse into sub-partition
        if (pivot_index <= k) {
            left = pivot_index + 1;
        }
        if (k <= pivot_index) {
            right = pivot_index - 1;
        }
    }
}

void floyd_rivest_select_position_index_loop(
    float *array, ptrdiff_t left, ptrdiff_t right, ptrdiff_t k,
    PositionAndIndexData<2, float, uint32_t> &data) {

    using std::iter_swap;

    while (right > left) {
        auto size = right - left;

        if (size > 600) {
            auto n = right - left + 1;
            auto i = k - left + 1;

            double z = std::log(n);
            double s = 0.5 * std::exp(2 * z / 3);
            double sd = 0.5 * std::sqrt(z * s * (n - s) / n);

            // We alternate choosing left and right pivots
            if (i < n / 2) {
                sd *= -1.0;
            }
            auto new_left = std::max(left, static_cast<ptrdiff_t>(k - i * s / n + sd));
            auto new_right = std::min(right, static_cast<ptrdiff_t>(k + (n - i) * s / n + sd));
            floyd_rivest_select_position_index_loop(array, new_left, new_right, k, data);
        }

        // Load the chosen pivot element
        auto t = array[k];

        // Place chosen pivot at end
        iter_swap(array + right, array + k);
        data.swap_elements(right, k);

        ptrdiff_t pivot_index = partition_vectorized_8(array, left, right, t, data);

        iter_swap(array + pivot_index, array + right);
        data.swap_elements(pivot_index, right);

        // Recurse into sub-partition
        if (pivot_index <= k) {
            left = pivot_index + 1;
        }
        if (k <= pivot_index) {
            right = pivot_index - 1;
        }
    }
}

} // namespace

namespace wenda {
namespace kdtree {
namespace detail {
size_t partition_float_array(float *array, size_t n, float pivot) {
    DummyData dummy;
    return partition_vectorized_8(array, 0, n, pivot, dummy);
}

template <typename T> T const &median_of_3(T const &a, T const &b, T const &c) {
    if ((b < a) ^ (c < a)) {
        return a;
    } else if ((b < a) ^ (b < c)) {
        return b;
    } else {
        return c;
    }
}

void quickselect_float_array(float *array, size_t n, size_t k) {
    size_t left = 0;
    size_t right = n;

    typedef r123::Philox4x32 RNG;
    RNG rng;

    RNG::ctr_type ctr = {{}};
    RNG::ukey_type ukey = {{}};

    uint32_t iteration = 0;

    // default pivot policy
    bool set_pivot_median_of_3 = true;

    float pivot;

    while (true) {
        if (right - left < 2) {
            return;
        }

        if (right - left == 2) {
            if (array[left] > array[right - 1]) {
                std::swap(array[left], array[right - 1]);
            }
            return;
        }

        size_t idx1, idx2, idx3;

        if (right - left == 3) {
            idx1 = left;
            idx2 = left + 1;
            idx3 = left + 2;
        } else {
            ctr.v[0] = iteration;
            auto r = rng(ctr, ukey);
            size_t n_range = right - left;
            size_t n_range_half = n_range / 2;
            size_t n_range_quarter = n_range / 4;

            idx1 = left + (r[0] % n_range_quarter);
            idx2 = left + (r[1] % n_range_half) + n_range_quarter;
            idx3 =
                left + (r[2] % (n_range_half - n_range_quarter)) + n_range_half + n_range_quarter;
        }

        if (set_pivot_median_of_3) {
            pivot = median_of_3(array[idx1], array[idx2], array[idx3]);
        } else {
            set_pivot_median_of_3 = true;
        }

        float smallest = std::numeric_limits<float>::max();
        float largest = std::numeric_limits<float>::lowest();
        auto pivot_index = partition_vectorized_8(array, left, right, pivot, smallest, largest);

        if (k < pivot_index) {
            if (pivot == smallest) {
                return;
            }

            right = pivot_index;
        } else {
            if (pivot == largest) {
                return;
            }

            if (left == pivot_index) {
                // we have stalled, so we need to set a pivot slightly above the current value
                // This only happens with a large number of equal values.
                set_pivot_median_of_3 = false;
                pivot = std::nextafter(pivot, std::numeric_limits<float>::max());
            }

            left = pivot_index;
        }
    }
}

void floyd_rivest_float_array(float *array, size_t n, size_t k) {
    floyd_rivest_select_float_loop(array, 0, n - 1, k);
}

void floyd_rivest_select_loop_position_array_avx2(
    PositionAndIndexArray<3> &array, std::ptrdiff_t left, std::ptrdiff_t right, std::ptrdiff_t k,
    int dimension) {

    std::array<float *, 2> positions_not_dimension;

    {
        size_t j = 0;
        for (size_t i = 0; i < 3; ++i) {
            if (i == dimension) {
                continue;
            }
            positions_not_dimension[j++] = array.positions_[i];
        }
    }

    PositionAndIndexData<2, float, uint32_t> data{
        dimension, positions_not_dimension, array.indices_.data()};

    floyd_rivest_select_position_index_loop(array.positions_[dimension], left, right, k, data);
}

void floyd_rivest_select_loop_position_array(
    PositionAndIndexArray<3> &array, std::ptrdiff_t left, std::ptrdiff_t right, std::ptrdiff_t k,
    int dimension) {

    auto &positions = array.positions_;

    auto comp = [&, dimension](ptrdiff_t l, ptrdiff_t r) {
        return positions[dimension][l] < positions[dimension][r];
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
            std::ptrdiff_t new_left =
                std::max(left, static_cast<std::ptrdiff_t>(k - i * s / n + sd));
            std::ptrdiff_t new_right =
                std::min(right, static_cast<std::ptrdiff_t>(k + (n - i) * s / n + sd));
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

} // namespace detail

} // namespace kdtree

} // namespace wenda

namespace {

// automatically generated permutation masks for partitioning an AVX2 vector.
const __m256i partition_permutation_masks[256] = {
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0),
    _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 7, 1), _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1),
    _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 7, 2), _mm256_setr_epi32(1, 3, 4, 5, 6, 7, 0, 2),
    _mm256_setr_epi32(0, 3, 4, 5, 6, 7, 1, 2), _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2),
    _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 3), _mm256_setr_epi32(1, 2, 4, 5, 6, 7, 0, 3),
    _mm256_setr_epi32(0, 2, 4, 5, 6, 7, 1, 3), _mm256_setr_epi32(2, 4, 5, 6, 7, 0, 1, 3),
    _mm256_setr_epi32(0, 1, 4, 5, 6, 7, 2, 3), _mm256_setr_epi32(1, 4, 5, 6, 7, 0, 2, 3),
    _mm256_setr_epi32(0, 4, 5, 6, 7, 1, 2, 3), _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3),
    _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 7, 4), _mm256_setr_epi32(1, 2, 3, 5, 6, 7, 0, 4),
    _mm256_setr_epi32(0, 2, 3, 5, 6, 7, 1, 4), _mm256_setr_epi32(2, 3, 5, 6, 7, 0, 1, 4),
    _mm256_setr_epi32(0, 1, 3, 5, 6, 7, 2, 4), _mm256_setr_epi32(1, 3, 5, 6, 7, 0, 2, 4),
    _mm256_setr_epi32(0, 3, 5, 6, 7, 1, 2, 4), _mm256_setr_epi32(3, 5, 6, 7, 0, 1, 2, 4),
    _mm256_setr_epi32(0, 1, 2, 5, 6, 7, 3, 4), _mm256_setr_epi32(1, 2, 5, 6, 7, 0, 3, 4),
    _mm256_setr_epi32(0, 2, 5, 6, 7, 1, 3, 4), _mm256_setr_epi32(2, 5, 6, 7, 0, 1, 3, 4),
    _mm256_setr_epi32(0, 1, 5, 6, 7, 2, 3, 4), _mm256_setr_epi32(1, 5, 6, 7, 0, 2, 3, 4),
    _mm256_setr_epi32(0, 5, 6, 7, 1, 2, 3, 4), _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 7, 5), _mm256_setr_epi32(1, 2, 3, 4, 6, 7, 0, 5),
    _mm256_setr_epi32(0, 2, 3, 4, 6, 7, 1, 5), _mm256_setr_epi32(2, 3, 4, 6, 7, 0, 1, 5),
    _mm256_setr_epi32(0, 1, 3, 4, 6, 7, 2, 5), _mm256_setr_epi32(1, 3, 4, 6, 7, 0, 2, 5),
    _mm256_setr_epi32(0, 3, 4, 6, 7, 1, 2, 5), _mm256_setr_epi32(3, 4, 6, 7, 0, 1, 2, 5),
    _mm256_setr_epi32(0, 1, 2, 4, 6, 7, 3, 5), _mm256_setr_epi32(1, 2, 4, 6, 7, 0, 3, 5),
    _mm256_setr_epi32(0, 2, 4, 6, 7, 1, 3, 5), _mm256_setr_epi32(2, 4, 6, 7, 0, 1, 3, 5),
    _mm256_setr_epi32(0, 1, 4, 6, 7, 2, 3, 5), _mm256_setr_epi32(1, 4, 6, 7, 0, 2, 3, 5),
    _mm256_setr_epi32(0, 4, 6, 7, 1, 2, 3, 5), _mm256_setr_epi32(4, 6, 7, 0, 1, 2, 3, 5),
    _mm256_setr_epi32(0, 1, 2, 3, 6, 7, 4, 5), _mm256_setr_epi32(1, 2, 3, 6, 7, 0, 4, 5),
    _mm256_setr_epi32(0, 2, 3, 6, 7, 1, 4, 5), _mm256_setr_epi32(2, 3, 6, 7, 0, 1, 4, 5),
    _mm256_setr_epi32(0, 1, 3, 6, 7, 2, 4, 5), _mm256_setr_epi32(1, 3, 6, 7, 0, 2, 4, 5),
    _mm256_setr_epi32(0, 3, 6, 7, 1, 2, 4, 5), _mm256_setr_epi32(3, 6, 7, 0, 1, 2, 4, 5),
    _mm256_setr_epi32(0, 1, 2, 6, 7, 3, 4, 5), _mm256_setr_epi32(1, 2, 6, 7, 0, 3, 4, 5),
    _mm256_setr_epi32(0, 2, 6, 7, 1, 3, 4, 5), _mm256_setr_epi32(2, 6, 7, 0, 1, 3, 4, 5),
    _mm256_setr_epi32(0, 1, 6, 7, 2, 3, 4, 5), _mm256_setr_epi32(1, 6, 7, 0, 2, 3, 4, 5),
    _mm256_setr_epi32(0, 6, 7, 1, 2, 3, 4, 5), _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 7, 6), _mm256_setr_epi32(1, 2, 3, 4, 5, 7, 0, 6),
    _mm256_setr_epi32(0, 2, 3, 4, 5, 7, 1, 6), _mm256_setr_epi32(2, 3, 4, 5, 7, 0, 1, 6),
    _mm256_setr_epi32(0, 1, 3, 4, 5, 7, 2, 6), _mm256_setr_epi32(1, 3, 4, 5, 7, 0, 2, 6),
    _mm256_setr_epi32(0, 3, 4, 5, 7, 1, 2, 6), _mm256_setr_epi32(3, 4, 5, 7, 0, 1, 2, 6),
    _mm256_setr_epi32(0, 1, 2, 4, 5, 7, 3, 6), _mm256_setr_epi32(1, 2, 4, 5, 7, 0, 3, 6),
    _mm256_setr_epi32(0, 2, 4, 5, 7, 1, 3, 6), _mm256_setr_epi32(2, 4, 5, 7, 0, 1, 3, 6),
    _mm256_setr_epi32(0, 1, 4, 5, 7, 2, 3, 6), _mm256_setr_epi32(1, 4, 5, 7, 0, 2, 3, 6),
    _mm256_setr_epi32(0, 4, 5, 7, 1, 2, 3, 6), _mm256_setr_epi32(4, 5, 7, 0, 1, 2, 3, 6),
    _mm256_setr_epi32(0, 1, 2, 3, 5, 7, 4, 6), _mm256_setr_epi32(1, 2, 3, 5, 7, 0, 4, 6),
    _mm256_setr_epi32(0, 2, 3, 5, 7, 1, 4, 6), _mm256_setr_epi32(2, 3, 5, 7, 0, 1, 4, 6),
    _mm256_setr_epi32(0, 1, 3, 5, 7, 2, 4, 6), _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6),
    _mm256_setr_epi32(0, 3, 5, 7, 1, 2, 4, 6), _mm256_setr_epi32(3, 5, 7, 0, 1, 2, 4, 6),
    _mm256_setr_epi32(0, 1, 2, 5, 7, 3, 4, 6), _mm256_setr_epi32(1, 2, 5, 7, 0, 3, 4, 6),
    _mm256_setr_epi32(0, 2, 5, 7, 1, 3, 4, 6), _mm256_setr_epi32(2, 5, 7, 0, 1, 3, 4, 6),
    _mm256_setr_epi32(0, 1, 5, 7, 2, 3, 4, 6), _mm256_setr_epi32(1, 5, 7, 0, 2, 3, 4, 6),
    _mm256_setr_epi32(0, 5, 7, 1, 2, 3, 4, 6), _mm256_setr_epi32(5, 7, 0, 1, 2, 3, 4, 6),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 7, 5, 6), _mm256_setr_epi32(1, 2, 3, 4, 7, 0, 5, 6),
    _mm256_setr_epi32(0, 2, 3, 4, 7, 1, 5, 6), _mm256_setr_epi32(2, 3, 4, 7, 0, 1, 5, 6),
    _mm256_setr_epi32(0, 1, 3, 4, 7, 2, 5, 6), _mm256_setr_epi32(1, 3, 4, 7, 0, 2, 5, 6),
    _mm256_setr_epi32(0, 3, 4, 7, 1, 2, 5, 6), _mm256_setr_epi32(3, 4, 7, 0, 1, 2, 5, 6),
    _mm256_setr_epi32(0, 1, 2, 4, 7, 3, 5, 6), _mm256_setr_epi32(1, 2, 4, 7, 0, 3, 5, 6),
    _mm256_setr_epi32(0, 2, 4, 7, 1, 3, 5, 6), _mm256_setr_epi32(2, 4, 7, 0, 1, 3, 5, 6),
    _mm256_setr_epi32(0, 1, 4, 7, 2, 3, 5, 6), _mm256_setr_epi32(1, 4, 7, 0, 2, 3, 5, 6),
    _mm256_setr_epi32(0, 4, 7, 1, 2, 3, 5, 6), _mm256_setr_epi32(4, 7, 0, 1, 2, 3, 5, 6),
    _mm256_setr_epi32(0, 1, 2, 3, 7, 4, 5, 6), _mm256_setr_epi32(1, 2, 3, 7, 0, 4, 5, 6),
    _mm256_setr_epi32(0, 2, 3, 7, 1, 4, 5, 6), _mm256_setr_epi32(2, 3, 7, 0, 1, 4, 5, 6),
    _mm256_setr_epi32(0, 1, 3, 7, 2, 4, 5, 6), _mm256_setr_epi32(1, 3, 7, 0, 2, 4, 5, 6),
    _mm256_setr_epi32(0, 3, 7, 1, 2, 4, 5, 6), _mm256_setr_epi32(3, 7, 0, 1, 2, 4, 5, 6),
    _mm256_setr_epi32(0, 1, 2, 7, 3, 4, 5, 6), _mm256_setr_epi32(1, 2, 7, 0, 3, 4, 5, 6),
    _mm256_setr_epi32(0, 2, 7, 1, 3, 4, 5, 6), _mm256_setr_epi32(2, 7, 0, 1, 3, 4, 5, 6),
    _mm256_setr_epi32(0, 1, 7, 2, 3, 4, 5, 6), _mm256_setr_epi32(1, 7, 0, 2, 3, 4, 5, 6),
    _mm256_setr_epi32(0, 7, 1, 2, 3, 4, 5, 6), _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 0, 7),
    _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 1, 7), _mm256_setr_epi32(2, 3, 4, 5, 6, 0, 1, 7),
    _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 2, 7), _mm256_setr_epi32(1, 3, 4, 5, 6, 0, 2, 7),
    _mm256_setr_epi32(0, 3, 4, 5, 6, 1, 2, 7), _mm256_setr_epi32(3, 4, 5, 6, 0, 1, 2, 7),
    _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7), _mm256_setr_epi32(1, 2, 4, 5, 6, 0, 3, 7),
    _mm256_setr_epi32(0, 2, 4, 5, 6, 1, 3, 7), _mm256_setr_epi32(2, 4, 5, 6, 0, 1, 3, 7),
    _mm256_setr_epi32(0, 1, 4, 5, 6, 2, 3, 7), _mm256_setr_epi32(1, 4, 5, 6, 0, 2, 3, 7),
    _mm256_setr_epi32(0, 4, 5, 6, 1, 2, 3, 7), _mm256_setr_epi32(4, 5, 6, 0, 1, 2, 3, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 4, 7), _mm256_setr_epi32(1, 2, 3, 5, 6, 0, 4, 7),
    _mm256_setr_epi32(0, 2, 3, 5, 6, 1, 4, 7), _mm256_setr_epi32(2, 3, 5, 6, 0, 1, 4, 7),
    _mm256_setr_epi32(0, 1, 3, 5, 6, 2, 4, 7), _mm256_setr_epi32(1, 3, 5, 6, 0, 2, 4, 7),
    _mm256_setr_epi32(0, 3, 5, 6, 1, 2, 4, 7), _mm256_setr_epi32(3, 5, 6, 0, 1, 2, 4, 7),
    _mm256_setr_epi32(0, 1, 2, 5, 6, 3, 4, 7), _mm256_setr_epi32(1, 2, 5, 6, 0, 3, 4, 7),
    _mm256_setr_epi32(0, 2, 5, 6, 1, 3, 4, 7), _mm256_setr_epi32(2, 5, 6, 0, 1, 3, 4, 7),
    _mm256_setr_epi32(0, 1, 5, 6, 2, 3, 4, 7), _mm256_setr_epi32(1, 5, 6, 0, 2, 3, 4, 7),
    _mm256_setr_epi32(0, 5, 6, 1, 2, 3, 4, 7), _mm256_setr_epi32(5, 6, 0, 1, 2, 3, 4, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 5, 7), _mm256_setr_epi32(1, 2, 3, 4, 6, 0, 5, 7),
    _mm256_setr_epi32(0, 2, 3, 4, 6, 1, 5, 7), _mm256_setr_epi32(2, 3, 4, 6, 0, 1, 5, 7),
    _mm256_setr_epi32(0, 1, 3, 4, 6, 2, 5, 7), _mm256_setr_epi32(1, 3, 4, 6, 0, 2, 5, 7),
    _mm256_setr_epi32(0, 3, 4, 6, 1, 2, 5, 7), _mm256_setr_epi32(3, 4, 6, 0, 1, 2, 5, 7),
    _mm256_setr_epi32(0, 1, 2, 4, 6, 3, 5, 7), _mm256_setr_epi32(1, 2, 4, 6, 0, 3, 5, 7),
    _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7), _mm256_setr_epi32(2, 4, 6, 0, 1, 3, 5, 7),
    _mm256_setr_epi32(0, 1, 4, 6, 2, 3, 5, 7), _mm256_setr_epi32(1, 4, 6, 0, 2, 3, 5, 7),
    _mm256_setr_epi32(0, 4, 6, 1, 2, 3, 5, 7), _mm256_setr_epi32(4, 6, 0, 1, 2, 3, 5, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 6, 4, 5, 7), _mm256_setr_epi32(1, 2, 3, 6, 0, 4, 5, 7),
    _mm256_setr_epi32(0, 2, 3, 6, 1, 4, 5, 7), _mm256_setr_epi32(2, 3, 6, 0, 1, 4, 5, 7),
    _mm256_setr_epi32(0, 1, 3, 6, 2, 4, 5, 7), _mm256_setr_epi32(1, 3, 6, 0, 2, 4, 5, 7),
    _mm256_setr_epi32(0, 3, 6, 1, 2, 4, 5, 7), _mm256_setr_epi32(3, 6, 0, 1, 2, 4, 5, 7),
    _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7), _mm256_setr_epi32(1, 2, 6, 0, 3, 4, 5, 7),
    _mm256_setr_epi32(0, 2, 6, 1, 3, 4, 5, 7), _mm256_setr_epi32(2, 6, 0, 1, 3, 4, 5, 7),
    _mm256_setr_epi32(0, 1, 6, 2, 3, 4, 5, 7), _mm256_setr_epi32(1, 6, 0, 2, 3, 4, 5, 7),
    _mm256_setr_epi32(0, 6, 1, 2, 3, 4, 5, 7), _mm256_setr_epi32(6, 0, 1, 2, 3, 4, 5, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(1, 2, 3, 4, 5, 0, 6, 7),
    _mm256_setr_epi32(0, 2, 3, 4, 5, 1, 6, 7), _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7),
    _mm256_setr_epi32(0, 1, 3, 4, 5, 2, 6, 7), _mm256_setr_epi32(1, 3, 4, 5, 0, 2, 6, 7),
    _mm256_setr_epi32(0, 3, 4, 5, 1, 2, 6, 7), _mm256_setr_epi32(3, 4, 5, 0, 1, 2, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 4, 5, 3, 6, 7), _mm256_setr_epi32(1, 2, 4, 5, 0, 3, 6, 7),
    _mm256_setr_epi32(0, 2, 4, 5, 1, 3, 6, 7), _mm256_setr_epi32(2, 4, 5, 0, 1, 3, 6, 7),
    _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7), _mm256_setr_epi32(1, 4, 5, 0, 2, 3, 6, 7),
    _mm256_setr_epi32(0, 4, 5, 1, 2, 3, 6, 7), _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 5, 4, 6, 7), _mm256_setr_epi32(1, 2, 3, 5, 0, 4, 6, 7),
    _mm256_setr_epi32(0, 2, 3, 5, 1, 4, 6, 7), _mm256_setr_epi32(2, 3, 5, 0, 1, 4, 6, 7),
    _mm256_setr_epi32(0, 1, 3, 5, 2, 4, 6, 7), _mm256_setr_epi32(1, 3, 5, 0, 2, 4, 6, 7),
    _mm256_setr_epi32(0, 3, 5, 1, 2, 4, 6, 7), _mm256_setr_epi32(3, 5, 0, 1, 2, 4, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 5, 3, 4, 6, 7), _mm256_setr_epi32(1, 2, 5, 0, 3, 4, 6, 7),
    _mm256_setr_epi32(0, 2, 5, 1, 3, 4, 6, 7), _mm256_setr_epi32(2, 5, 0, 1, 3, 4, 6, 7),
    _mm256_setr_epi32(0, 1, 5, 2, 3, 4, 6, 7), _mm256_setr_epi32(1, 5, 0, 2, 3, 4, 6, 7),
    _mm256_setr_epi32(0, 5, 1, 2, 3, 4, 6, 7), _mm256_setr_epi32(5, 0, 1, 2, 3, 4, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(1, 2, 3, 4, 0, 5, 6, 7),
    _mm256_setr_epi32(0, 2, 3, 4, 1, 5, 6, 7), _mm256_setr_epi32(2, 3, 4, 0, 1, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 3, 4, 2, 5, 6, 7), _mm256_setr_epi32(1, 3, 4, 0, 2, 5, 6, 7),
    _mm256_setr_epi32(0, 3, 4, 1, 2, 5, 6, 7), _mm256_setr_epi32(3, 4, 0, 1, 2, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 4, 3, 5, 6, 7), _mm256_setr_epi32(1, 2, 4, 0, 3, 5, 6, 7),
    _mm256_setr_epi32(0, 2, 4, 1, 3, 5, 6, 7), _mm256_setr_epi32(2, 4, 0, 1, 3, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 4, 2, 3, 5, 6, 7), _mm256_setr_epi32(1, 4, 0, 2, 3, 5, 6, 7),
    _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7), _mm256_setr_epi32(4, 0, 1, 2, 3, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7),
    _mm256_setr_epi32(0, 2, 3, 1, 4, 5, 6, 7), _mm256_setr_epi32(2, 3, 0, 1, 4, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 3, 2, 4, 5, 6, 7), _mm256_setr_epi32(1, 3, 0, 2, 4, 5, 6, 7),
    _mm256_setr_epi32(0, 3, 1, 2, 4, 5, 6, 7), _mm256_setr_epi32(3, 0, 1, 2, 4, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(1, 2, 0, 3, 4, 5, 6, 7),
    _mm256_setr_epi32(0, 2, 1, 3, 4, 5, 6, 7), _mm256_setr_epi32(2, 0, 1, 3, 4, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(1, 0, 2, 3, 4, 5, 6, 7),
    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)};
} // namespace