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

template <typename DistanceT, typename QueueT> struct InsertShorterDistanceVanilla {
    typedef DistanceT distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    void operator()(
        tcb::span<const PositionAndIndex> const &positions, std::array<float, 3> const &query,
        QueueT &distances, DistanceT const &distance_func) {

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
template <typename DistanceT, typename QueueT, int Unroll=4> struct InsertShorterDistanceUnrolled {
    typedef DistanceT distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    void operator()(
        tcb::span<const PositionAndIndex> positions, std::array<float, 3> const &query,
        QueueT &distances, DistanceT const &distance) const {
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

template<typename DistanceT, typename QueueT>
using InsertShorterDistanceUnrolled4 = InsertShorterDistanceUnrolled<DistanceT, QueueT, 4>;

template<typename DistanceT, typename QueueT>
using InsertShorterDistanceUnrolled8 = InsertShorterDistanceUnrolled<DistanceT, QueueT, 8>;

template <typename DistanceT, typename QueueT>
struct InsertShorterDistanceAVX : InsertShorterDistanceUnrolled<DistanceT, QueueT, 4> {};

template <typename QueueT> struct InsertShorterDistanceAVX<L2Distance, QueueT> {
    typedef L2Distance distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    void operator()(
        tcb::span<const PositionAndIndex> positions, std::array<float, 3> const &query,
        QueueT &distances, L2Distance const &distance_func) const {

        // Load in the query vector, duplicated across the upper and lower lanes.
        __m128 q_half =
            _mm_maskload_ps(query.data(), _mm_set_epi32(0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
        __m256 q = _mm256_set_m128(q_half, q_half);

        // Buffer holding the loaded positions + indices
        // Each register can store 2 PositionAndIndex instances.
        __m256i v0, v1, v2, v3;

        // Buffer holding the indices for this iteration of the unrolled loop.
        // Note that this buffer, along with the buffer below, are not in order compared
        // to loop iteration order. However, their respective order is the same.
        uint32_t indices_buffer[8];
        // Buffer holding computed distances for this iteration of the unrolled loop.
        float distances_buffer[8];

        int i = 0;
        int num_points = static_cast<int>(positions.size());

        __m256 sum;

        float current_best_distance = distances.top().first;

        PositionAndIndex const *positions_ptr = positions.data();

        // main unrolled loop.
        for (; i < num_points - 7; i += 8) {
            _mm_prefetch(positions_ptr + i + 8, _MM_HINT_NTA);
            v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(positions_ptr + i));
            v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(positions_ptr + i + 2));
            v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(positions_ptr + i + 4));
            v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(positions_ptr + i + 6));

            assert(i + 7 < num_points);

            // prefetch next loop
            //_mm_prefetch(reinterpret_cast<const char *>(positions_ptr + i + 8), _MM_HINT_NTA);
            //_mm_prefetch(reinterpret_cast<const char *>(positions_ptr + i + 12), _MM_HINT_NTA);

            // extract indices from packed data.
            // Note that indices are not extracted in the original order, but rather
            // in the following order: 0, 2, 4, 6, 1, 3, 5, 7.
            // However, this also corresponds to the order in which the dot products in sum are
            // computed, hence there is no further adjustment required.
            _mm256_storeu_si256(
                reinterpret_cast<__m256i *>(indices_buffer),
                _mm256_unpackhi_epi32(
                    _mm256_unpackhi_epi32(v0, v2), _mm256_unpackhi_epi32(v1, v3)));

            // Accumulate squared distance for each position.
            // By using the dot product instruction with careful selection of the output
            // we can accumulate the distances in each segment of the vector register.
            sum = dot_product<0>(_mm256_castsi256_ps(v0), q);
            sum = _mm256_add_ps(sum, dot_product<1>(_mm256_castsi256_ps(v1), q));
            sum = _mm256_add_ps(sum, dot_product<2>(_mm256_castsi256_ps(v2), q));
            sum = _mm256_add_ps(sum, dot_product<3>(_mm256_castsi256_ps(v3), q));

            // Evaluate whether we need to update the best distance.
            __m256 cmp =
                _mm256_cmp_ps(sum, _mm256_broadcast_ss(&current_best_distance), _CMP_LT_OQ);
            int mask = _mm256_movemask_ps(cmp);

            if (mask == 0) {
                // early-exit when no comparison are triggered
                continue;
            }

            // Extract distances into stack buffer for use by scalar code.
            _mm256_storeu_ps(distances_buffer, sum);

            for (int j = 0; j < 8; ++j) {
                if (distances_buffer[j] >= current_best_distance) {
                    continue;
                }

                distances.replace_top({distances_buffer[j], indices_buffer[j]});
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

            distances.replace_top({distance, positions[i].index});
            current_best_distance = distances.top().first;
        }
    }
};

} // namespace

} // namespace kdtree
} // namespace wenda
