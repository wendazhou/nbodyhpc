#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "kdtree_opt.hpp"
#include "utils.hpp"
#include "tournament_tree.hpp"

using namespace wenda::kdtree;


namespace {


template<template<typename, typename> typename Inserter, typename DistanceT, typename QueueT>
std::vector<std::pair<float, uint32_t>> find_nearest_inserter(std::vector<std::array<float, 3>> const& positions, std::array<float, 3> const& query, int k, DistanceT const& distance) {
    std::vector<wenda::kdtree::PositionAndIndex> position_and_indices = wenda::kdtree::make_position_and_indices(positions);

    QueueT queue(k, {std::numeric_limits<float>::max(), -1});

    Inserter<DistanceT, QueueT> inserter;

    inserter(position_and_indices, query, queue, distance);

    std::vector<std::pair<float, uint32_t>> result(k);
    queue.copy_values(result.begin());
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

typedef PriorityQueue<std::pair<float, uint32_t>, PairLessFirst> PairPriorityQueue;
typedef TournamentTree<std::pair<float, uint32_t>, PairLessFirst> PairTournamentTree;


TEST(OptInsertionTest, UnrolledInsertion) {
    auto positions = fill_random_positions<float, 3>(128, 42);
    auto queries = fill_random_positions<float, 3>(4, 43);

    for (auto const& query : queries) {
        auto result_vanilla = find_nearest_inserter<InsertShorterDistanceVanilla, L2Distance, PairPriorityQueue>(positions, query, 8, L2Distance());
        auto result_unrolled = find_nearest_inserter<InsertShorterDistanceUnrolled4, L2Distance, PairPriorityQueue>(positions, query, 8, L2Distance());

        ASSERT_EQ(result_vanilla, result_unrolled);
    }
}

TEST(OptInsertionTest, UnrolledInsertionTournamentTree) {
    auto positions = fill_random_positions<float, 3>(128, 42);
    auto queries = fill_random_positions<float, 3>(4, 43);

    for (auto const& query : queries) {
        auto result_vanilla = find_nearest_inserter<InsertShorterDistanceVanilla, L2Distance, PairPriorityQueue>(positions, query, 8, L2Distance());
        auto result_unrolled = find_nearest_inserter<InsertShorterDistanceUnrolled4, L2Distance, PairTournamentTree>(positions, query, 8, L2Distance());

        ASSERT_EQ(result_vanilla, result_unrolled);
    }
}

TEST(OptInsertionTest, Avx2Insertion) {
    auto positions = fill_random_positions<float, 3>(128, 42);
    auto queries = fill_random_positions<float, 3>(4, 43);

    for (auto const& query : queries) {
        auto result_vanilla = find_nearest_inserter<InsertShorterDistanceVanilla, L2Distance, PairPriorityQueue>(positions, query, 8, L2Distance());
        auto result_avx = find_nearest_inserter<InsertShorterDistanceAVX, L2Distance, PairPriorityQueue>(positions, query, 8, L2Distance());

        ASSERT_EQ(result_vanilla, result_avx);
    }
}

TEST(OptInsertionTest, Avx2InsertionTournamentTree) {
    auto positions = fill_random_positions<float, 3>(128, 42);
    auto queries = fill_random_positions<float, 3>(4, 43);

    for (auto const& query : queries) {
        auto result_vanilla = find_nearest_inserter<InsertShorterDistanceVanilla, L2Distance, PairPriorityQueue>(positions, query, 8, L2Distance());
        auto result_avx = find_nearest_inserter<InsertShorterDistanceAVX, L2Distance, PairTournamentTree>(positions, query, 8, L2Distance());

        ASSERT_EQ(result_vanilla, result_avx);
    }
}

class TournamentTreeFixture : public ::testing::TestWithParam<int> {};

TEST_P(TournamentTreeFixture, TournamentTreeTest) {
    int num_max = GetParam();

    std::vector<int> values(1000);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    std::generate(values.begin(), values.end(), [&]() { return dist(rng); });

    wenda::kdtree::TournamentTree<int> tree(num_max, std::numeric_limits<int>::max());

    for (auto const& value : values) {
        if (value < tree.top())  {
            tree.replace_top(value);
        }
    }

    std::vector<int> result;
    tree.copy_values(std::back_inserter(result));

    ASSERT_EQ(result.size(), num_max);

    std::vector<int> expected_result(num_max);
    std::nth_element(values.begin(), values.begin() + num_max, values.end());
    std::copy(values.begin(), values.begin() + num_max, expected_result.begin());

    std::sort(result.begin(), result.end());
    std::sort(expected_result.begin(), expected_result.end());

    ASSERT_EQ(result, expected_result);
}


INSTANTIATE_TEST_SUITE_P(
    TournamentTreePriority, TournamentTreeFixture, testing::Values(2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 23));



