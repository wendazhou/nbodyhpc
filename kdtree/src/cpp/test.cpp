#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include "kdtree.hpp"

namespace {

float l2_distance_squared(std::array<float, 3> const &left, std::array<float, 3> const &right) {
    float result = 0;

    for (size_t i = 0; i < 3; ++i) {
        result += (left[i] - right[i]) * (left[i] - right[i]);
    }

    return result;
}

std::vector<std::array<float, 3>> fill_random_positions(int n, int64_t seed) {
    std::vector<std::array<float, 3>> positions(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        positions[i][0] = dist(rng);
        positions[i][1] = dist(rng);
        positions[i][2] = dist(rng);
    }

    return positions;
}

float find_nearest_naive(tcb::span<const std::array<float, 3>> positions, std::array<float, 3> const& query) {
    return std::transform_reduce(
        positions.begin(),
        positions.end(),
        std::numeric_limits<float>::max(),
        [](float a, float b) { return std::min(a, b); },
        [&](std::array<float, 3> const &x) { return l2_distance_squared(x, query); });
}

} // namespace


class KDTreeRandomTest : public ::testing::TestWithParam<int> {
};


// Demonstrate some basic assertions.
TEST_P(KDTreeRandomTest, BuildAndFindNearest) { 
    auto positions = fill_random_positions(GetParam(), 42);
    std::array<float, 3> query = {0.5,0.5,0.5};

    auto tree = wenda::kdtree::build_kdtree(positions);
    auto result = wenda::kdtree::query_kdtree(tree.get(), query);

    auto naive_result = find_nearest_naive(positions, query);

    ASSERT_FLOAT_EQ(result, naive_result);
}

INSTANTIATE_TEST_SUITE_P(BuildAndFindNearestSmall, KDTreeRandomTest, testing::Values(100, 1000, 10000));
