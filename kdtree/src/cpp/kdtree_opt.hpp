#pragma once

#include <queue>
#include <utility>

#include <immintrin.h>

#include "kdtree.hpp"

namespace wenda {
namespace kdtree {

namespace {

struct PairLessFirst {
    bool operator()(
        std::pair<float, uint32_t> const &left, std::pair<float, uint32_t> const &right) const {
        return left.first < right.first;
    }
};

template <int i> __m256 dot_product(__m256 v, __m256 q) {
    __m256 delta = _mm256_sub_ps(v, q);
    return _mm256_dp_ps(delta, delta, 0b01110000 + (1 << i));
}

template <typename DistanceT> struct InsertShorterDistanceVanilla {
    typedef std::pair<float, uint32_t> result_t;
    typedef std::priority_queue<result_t, std::vector<result_t>, PairLessFirst> queue_t;

    void operator()(
        tcb::span<const PositionAndIndex> const &positions, std::array<float, 3> const &query,
        queue_t &distances, DistanceT const &distance_func) {

        uint32_t num_points = static_cast<uint32_t>(positions.size());
        float current_best_distance = distances.top().first;

        for (uint32_t i = 0; i < num_points; ++i) {
            auto dist = distance_func(query, positions[i].position);

            if (dist >= current_best_distance) {
                continue;
            }

            distances.pop();
            distances.push({dist, positions[i].index});
            current_best_distance = distances.top().first;
        }
    }
};

//! Template for unrolled shorter distance insertion.
//! This implementation partially unrolls the inner loop, to give
//! the optimizer a better chance to vectorize it.
template <typename DistanceT, int Unroll> struct InsertShorterDistanceUnrolled {
    typedef std::pair<float, uint32_t> result_t;
    typedef std::priority_queue<result_t, std::vector<result_t>, PairLessFirst> queue_t;

    void operator()(
        tcb::span<const PositionAndIndex> positions, std::array<float, 3> const &query,
        queue_t &distances, DistanceT const &distance) const {
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

                distances.pop();
                distances.push({distances_buffer[j], indices_buffer[j]});
                current_best_distance = distances.top().first;
            }
        }

        if (Unroll > 1) {
            for (; i < num_points; ++i) {
                float dist = distance(positions[i].position, query);

                if (dist >= current_best_distance) {
                    continue;
                }

                distances.pop();
                distances.push({dist, positions[i].index});
                current_best_distance = distances.top().first;
            }
        }
    }
};

template <typename DistanceT> struct InsertShorterDistanceAVX;

template <> struct InsertShorterDistanceAVX<L2Distance> {
    typedef std::pair<float, uint32_t> result_t;
    typedef std::priority_queue<result_t, std::vector<result_t>, PairLessFirst> queue_t;

    void operator()(
        tcb::span<const PositionAndIndex> positions, std::array<float, 3> const &query,
        queue_t &distances, L2Distance const &distance_func) const {

        __m128 q_half =
            _mm_maskload_ps(query.data(), _mm_set_epi32(0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
        __m256 q = _mm256_set_m128(q_half, q_half);

        __m256 v[4];
        uint32_t indices_buffer[8];
        float distances_buffer[8];

        size_t i = 0;
        size_t num_points = positions.size();

        __m256 sum;

        float current_best_distance = distances.top().first;

        // main unrolled loop.
        for (; i < num_points + 1 - 8; i += 8) {
            for (int j = 0; j < 4; ++j) {
                v[j] =
                    _mm256_loadu_ps(reinterpret_cast<const float *>(positions.data() + i + 2 * j));
            }
            for (int j = 0; j < 8; ++j) {
                indices_buffer[j] = positions[i + j].index;
            }

            sum = dot_product<0>(v[0], q);
            sum = _mm256_add_ps(sum, dot_product<1>(v[1], q));
            sum = _mm256_add_ps(sum, dot_product<2>(v[2], q));
            sum = _mm256_add_ps(sum, dot_product<3>(v[3], q));

            sum = _mm256_permutevar8x32_ps(sum, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            __m256 cmp =
                _mm256_cmp_ps(sum, _mm256_broadcast_ss(&current_best_distance), _CMP_LT_OQ);
            int mask = _mm256_movemask_ps(cmp);

            if (mask == 0) {
                // early-exit when no comparison are triggered
                continue;
            }

            _mm256_storeu_ps(distances_buffer, sum);

            for (int j = 0; j < 8; ++j) {
                if (distances_buffer[j] >= current_best_distance) {
                    continue;
                }

                distances.pop();
                distances.push({distances_buffer[j], indices_buffer[j]});
                current_best_distance = distances.top().first;
            }
        }

        // tail loop.
        for (; i < num_points; ++i) {
            auto const &pos = positions[i].position;
            auto distance = distance_func(pos, query);

            if (distance >= current_best_distance) {
                continue;
            }

            distances.pop();
            distances.push({distance, positions[i].index});
            current_best_distance = distances.top().first;
        }
    }
};

} // namespace

} // namespace kdtree
} // namespace wenda
