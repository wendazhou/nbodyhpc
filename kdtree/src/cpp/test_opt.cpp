#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "kdtree_opt.hpp"
#include "utils.hpp"

using namespace wenda::kdtree;


namespace {


template<typename Inserter, typename DistanceT>
std::vector<std::pair<float, uint32_t>> find_nearest_inserter(std::vector<std::array<float, 3>> const& positions, std::array<float, 3> const& query, int k, DistanceT const& distance) {
    std::vector<wenda::kdtree::PositionAndIndex> position_and_indices = wenda::kdtree::make_position_and_indices(positions);

    typename Inserter::queue_t queue;
    for(int i = 0; i < k; ++i) {
        queue.push({std::numeric_limits<float>::max(), -1});
    }

    Inserter inserter;

    inserter(position_and_indices, query, queue, distance);

    auto result = wenda::kdtree::get_container_from_adapter(queue);
    std::sort(result.begin(), result.end());
    return result;
}


template <typename T, size_t R>
std::vector<std::array<T, R>> fill_random_positions(int n, unsigned int seed, float boxsize=1.0f) {
    std::vector<std::array<T, R>> positions(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(0.0f, boxsize);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < R; ++j) {
            positions[i][j] = dist(rng);
        }
    }

    return positions;
}

}


TEST(OptInsertionTest, UnrolledInsertion) {
    auto positions = fill_random_positions<float, 3>(128, 42);
    auto queries = fill_random_positions<float, 3>(4, 43);

    for (auto const& query : queries) {
        auto result_vanilla = find_nearest_inserter<InsertShorterDistanceVanilla<L2Distance>>(positions, query, 8, L2Distance());
        auto result_unrolled = find_nearest_inserter<InsertShorterDistanceUnrolled<L2Distance, 4>>(positions, query, 8, L2Distance());

        ASSERT_EQ(result_vanilla, result_unrolled);
    }
}

TEST(OptInsertionTest, Avx2Insertion) {
    auto positions = fill_random_positions<float, 3>(128, 42);
    auto queries = fill_random_positions<float, 3>(4, 43);

    for (auto const& query : queries) {
        auto result_vanilla = find_nearest_inserter<InsertShorterDistanceVanilla<L2Distance>>(positions, query, 8, L2Distance());
        auto result_avx = find_nearest_inserter<InsertShorterDistanceAVX<L2Distance>>(positions, query, 8, L2Distance());

        ASSERT_EQ(result_vanilla, result_avx);
    }
}



