#pragma once

#include <queue>
#include <utility>

#include <immintrin.h>

#include "kdtree.hpp"

namespace wenda {
namespace kdtree {

struct PairLessFirst {
    bool operator()(
        std::pair<float, uint32_t> const &left, std::pair<float, uint32_t> const &right) const {
        return left.first < right.first;
    }
};

template <typename DistanceT, typename QueueT> struct InsertShorterDistanceVanilla {
    typedef DistanceT distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    template <typename Container>
    void operator()(
        Container const &positions, std::array<float, 3> const &query, QueueT &distances,
        DistanceT const &distance_func) {

        uint32_t num_points = static_cast<uint32_t>(positions.size());
        float current_best_distance = distances.top().first;

        for (uint32_t i = 0; i < num_points; ++i) {
            auto dist = distance_func(query, positions[i].position);

            if (dist >= current_best_distance) {
                continue;
            }

            distances.replace_top({dist, positions[i].index});
            current_best_distance = distances.top().first;
        }
    }
};

//! Template for unrolled shorter distance insertion.
//! This implementation partially unrolls the inner loop, to give
//! the optimizer a better chance to vectorize it.
template <typename DistanceT, typename QueueT, int Unroll = 4>
struct InsertShorterDistanceUnrolled {
    typedef DistanceT distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    template <typename ContainerT>
    void operator()(
        ContainerT const &positions, std::array<float, 3> const &query, QueueT &distances,
        DistanceT const &distance) const {
        uint32_t num_points = static_cast<uint32_t>(positions.size());

        uint32_t i = 0;

        float distances_buffer[Unroll];
        uint32_t indices_buffer[Unroll];

        float current_best_distance = distances.top().first;

        for (; i < num_points + 1 - Unroll; i += Unroll) {
            for (int j = 0; j < Unroll; ++j) {
                distances_buffer[j] = distance(positions[i + j].position, query);
                indices_buffer[j] = positions[i + j].index;
            }

            for (int j = 0; j < Unroll; ++j) {
                if (distances_buffer[j] >= current_best_distance) {
                    continue;
                }

                distances.replace_top({distances_buffer[j], indices_buffer[j]});
                current_best_distance = distances.top().first;
            }
        }

        if (Unroll > 1) {
            for (; i < num_points; ++i) {
                float dist = distance(positions[i].position, query);

                if (dist >= current_best_distance) {
                    continue;
                }

                distances.replace_top({dist, positions[i].index});
                current_best_distance = distances.top().first;
            }
        }
    }
};

template <typename DistanceT, typename QueueT>
using InsertShorterDistanceUnrolled4 = InsertShorterDistanceUnrolled<DistanceT, QueueT, 4>;

template <typename DistanceT, typename QueueT>
using InsertShorterDistanceUnrolled8 = InsertShorterDistanceUnrolled<DistanceT, QueueT, 8>;

template <typename DistanceT, typename QueueT> struct InsertShorterDistanceAVX;

template <typename QueueT> struct InsertShorterDistanceAVX<L2Distance, QueueT> {
    typedef L2Distance distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    void operator()(
        OffsetRangeContainerWrapper<const PositionAndIndexArray<3>> const &positions,
        std::array<float, 3> const &query, queue_t &distances,
        distance_t const &distance_func) const {

        size_t num_points = positions.size();

        std::array<const float *, 3> positions_ptr;
        for (size_t i = 0; i < 3; ++i) {
            positions_ptr[i] = positions.container_.positions_[i] + positions.offset;
        }

        const uint32_t *indices_ptr = positions.container_.indices_.data() + positions.offset;

        __m256 qx = _mm256_set1_ps(query[0]);
        __m256 qy = _mm256_set1_ps(query[1]);
        __m256 qz = _mm256_set1_ps(query[2]);

        __m256 current_best_distance = _mm256_set1_ps(distances.top().first);

        alignas(32) float distances_buffer[8];

        for (size_t i = 0; i < num_points - 7; i += 8) {
            __m256 x = _mm256_load_ps(positions_ptr[0] + i);
            __m256 y = _mm256_load_ps(positions_ptr[1] + i);
            __m256 z = _mm256_load_ps(positions_ptr[2] + i);

            __m256 dx = _mm256_sub_ps(x, qx);
            __m256 dy = _mm256_sub_ps(y, qy);
            __m256 dz = _mm256_sub_ps(z, qz);

            __m256 dist = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz));

            __m256 cmp = _mm256_cmp_ps(dist, current_best_distance, _CMP_LT_OQ);
            uint32_t mask = _mm256_movemask_ps(cmp);

            if (mask == 0) {
                continue;
            }

            _mm256_store_ps(distances_buffer, dist);
            int subelement = 0;

            while (mask != 0) {
                if ((mask & 0x1) && (distances_buffer[subelement] < distances.top().first)) {
                    distances.replace_top(
                        {distances_buffer[subelement], indices_ptr[i + subelement]});
                    current_best_distance = _mm256_set1_ps(distances.top().first);
                }

                subelement += 1;
                mask = mask >> 1;
            }
        }
    }
};

namespace detail {

inline __m256 compute_distance_squared_box_1d(__m256 x, __m256 qx, __m256 box) {
    __m256 dx = _mm256_sub_ps(x, qx);
    __m256 dx1 = _mm256_add_ps(dx, box);
    __m256 dx2 = _mm256_sub_ps(dx, box);

    dx = _mm256_mul_ps(dx, dx);
    dx1 = _mm256_mul_ps(dx1, dx1);
    dx2 = _mm256_mul_ps(dx2, dx2);

    dx = _mm256_min_ps(dx, dx1);
    dx = _mm256_min_ps(dx, dx2);
    return dx;
}
} // namespace detail

template <typename QueueT> struct InsertShorterDistanceAVX<L2PeriodicDistance<float>, QueueT> {
    typedef L2PeriodicDistance<float> distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    void operator()(
        OffsetRangeContainerWrapper<const PositionAndIndexArray<3>> const &positions,
        std::array<float, 3> const &query, queue_t &distances,
        distance_t const &distance_func) const {

        size_t num_points = positions.size();

        std::array<const float *, 3> positions_ptr;
        for (size_t i = 0; i < 3; ++i) {
            positions_ptr[i] = positions.container_.positions_[i] + positions.offset;
        }

        const uint32_t *indices_ptr = positions.container_.indices_.data() + positions.offset;

        __m256 qx = _mm256_set1_ps(query[0]);
        __m256 qy = _mm256_set1_ps(query[1]);
        __m256 qz = _mm256_set1_ps(query[2]);

        __m256 current_best_distance = _mm256_set1_ps(distances.top().first);
        __m256 box_size = _mm256_set1_ps(distance_func.box_size_);

        alignas(32) float distances_buffer[8];

        for (size_t i = 0; i < num_points - 7; i += 8) {
            __m256 x = _mm256_load_ps(positions_ptr[0] + i);
            __m256 y = _mm256_load_ps(positions_ptr[1] + i);
            __m256 z = _mm256_load_ps(positions_ptr[2] + i);

            __m256 dx2 = detail::compute_distance_squared_box_1d(x, qx, box_size);
            __m256 dy2 = detail::compute_distance_squared_box_1d(y, qy, box_size);
            __m256 dz2 = detail::compute_distance_squared_box_1d(z, qz, box_size);

            __m256 dist = _mm256_add_ps(_mm256_add_ps(dx2, dy2), dz2);

            __m256 cmp = _mm256_cmp_ps(dist, current_best_distance, _CMP_LT_OQ);
            uint32_t mask = _mm256_movemask_ps(cmp);

            if (mask == 0) {
                continue;
            }

            _mm256_store_ps(distances_buffer, dist);
            int subelement = 0;

            while (mask != 0) {
                if (mask & 0x1) {
                    distances.replace_top(
                        {distances_buffer[subelement], indices_ptr[i + subelement]});
                    current_best_distance = _mm256_set1_ps(distances.top().first);
                }

                subelement += 1;
                mask = mask >> 1;
            }
        }
    }
};

} // namespace kdtree
} // namespace wenda
