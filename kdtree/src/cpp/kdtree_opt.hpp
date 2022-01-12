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

template <typename DistanceT, typename QueueT> struct InsertShorterDistanceVanilla {
    typedef DistanceT distance_t;
    typedef QueueT queue_t;
    typedef std::pair<float, uint32_t> result_t;

    template<typename Container>
    void operator()(
        Container const &positions, std::array<float, 3> const &query,
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

    template<typename ContainerT>
    void operator()(
        ContainerT const& positions, std::array<float, 3> const &query,
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


} // namespace

} // namespace kdtree
} // namespace wenda
