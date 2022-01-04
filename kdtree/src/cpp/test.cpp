#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "utils.hpp"

namespace {

float l2_distance_squared(std::array<float, 3> const &left, std::array<float, 3> const &right) {
    float result = 0;

    for (size_t i = 0; i < 3; ++i) {
        result += (left[i] - right[i]) * (left[i] - right[i]);
    }

    return result;
}

std::vector<std::array<float, 3>> fill_random_positions(int n, unsigned int seed) {
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

float find_nearest_naive(
    tcb::span<const std::array<float, 3>> positions, std::array<float, 3> const &query) {
    return std::transform_reduce(
        positions.begin(),
        positions.end(),
        std::numeric_limits<float>::max(),
        [](float a, float b) { return std::min(a, b); },
        [&](std::array<float, 3> const &x) { return l2_distance_squared(x, query); });
}

std::vector<std::pair<float, uint32_t>> find_nearest_naive(
    tcb::span<const std::array<float, 3>> positions, const std::array<float, 3> &query, int k) {
    std::priority_queue<std::pair<float, uint32_t>> distances(
        {}, std::vector<std::pair<float, uint32_t>>(k, {std::numeric_limits<float>::max(), -1}));

    for (uint32_t i = 0; i < positions.size(); ++i) {
        auto dist = l2_distance_squared(positions[i], query);
        if (dist < distances.top().first) {
            distances.pop();
            distances.push({dist, i});
        }
    }

    std::vector result(std::move(wenda::kdtree::get_container_from_adapter(distances)));
    std::sort(result.begin(), result.end());

    for (auto &p : result) {
        p.first = std::sqrt(p.first);
    }

    return result;
}

} // namespace

class KDTreeRandomTest : public ::testing::TestWithParam<int> {};

TEST_P(KDTreeRandomTest, BuildAndFindNearestClass) {
    auto positions = fill_random_positions(GetParam(), 42);
    std::array<float, 3> query = {0.5, 0.5, 0.5};

    auto tree = wenda::kdtree::KDTree(positions);
    wenda::kdtree::KDTreeQueryStatistics statistics;
    auto result = tree.find_closest(query, 4, wenda::kdtree::L2Distance{}, &statistics);

    auto naive_result = find_nearest_naive(positions, query, 4);

    ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));

    ASSERT_FLOAT_EQ(result[0].first, naive_result[0].first);
    ASSERT_FLOAT_EQ(result[1].first, naive_result[1].first);
    ASSERT_FLOAT_EQ(result[2].first, naive_result[2].first);
    ASSERT_FLOAT_EQ(result[3].first, naive_result[3].first);

    ASSERT_EQ(result[0].second, naive_result[0].second);
    ASSERT_EQ(result[1].second, naive_result[1].second);
    ASSERT_EQ(result[2].second, naive_result[2].second);
    ASSERT_EQ(result[3].second, naive_result[3].second);

    ASSERT_GT(statistics.nodes_pruned, 0);
    ASSERT_GT(statistics.nodes_visited, 0);
    ASSERT_LT(statistics.nodes_visited, positions.size());
}

TEST_P(KDTreeRandomTest, BuildAndFindNearestClassMT) {
    auto positions = fill_random_positions(GetParam(), 42);
    std::array<float, 3> query = {0.5, 0.5, 0.5};

    auto tree = wenda::kdtree::KDTree(positions, {.max_threads = -1});
    wenda::kdtree::KDTreeQueryStatistics statistics;
    auto result = tree.find_closest(query, 4, wenda::kdtree::L2Distance{}, &statistics);

    auto naive_result = find_nearest_naive(positions, query, 4);

    ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));

    ASSERT_FLOAT_EQ(result[0].first, naive_result[0].first);
    ASSERT_FLOAT_EQ(result[1].first, naive_result[1].first);
    ASSERT_FLOAT_EQ(result[2].first, naive_result[2].first);
    ASSERT_FLOAT_EQ(result[3].first, naive_result[3].first);

    ASSERT_EQ(result[0].second, naive_result[0].second);
    ASSERT_EQ(result[1].second, naive_result[1].second);
    ASSERT_EQ(result[2].second, naive_result[2].second);
    ASSERT_EQ(result[3].second, naive_result[3].second);

    ASSERT_GT(statistics.nodes_pruned, 0);
    ASSERT_GT(statistics.nodes_visited, 0);
    ASSERT_LT(statistics.nodes_visited, positions.size());
}

INSTANTIATE_TEST_SUITE_P(
    BuildAndFindNearestSmall, KDTreeRandomTest, testing::Values(100, 1000, 10000));

TEST(KDTreeMetric, TestL2PeriodicBox) {
    auto positions = fill_random_positions(100, 42);
    std::array<float, 6> box = {0.2f, 0.5f, 0.4f, 0.6f, 0.0f, 0.1f};

    wenda::kdtree::L2Distance distance_naive;
    wenda::kdtree::L2PeriodicDistance<float> distance_periodic{1.0f};

    for (auto const &p : positions) {
        auto dist = distance_periodic.box_distance(p, box);

        auto dist_naive = std::numeric_limits<float>::max();

        for (int i1 = 0; i1 < 3; ++i1) {
            for(int i2 = 0; i2 < 3; ++i2) {
                for(int i3 = 0; i3 < 3; ++i3) {
                    auto p_mod = p;
                    p_mod[0] += (i1 - 1);
                    p_mod[1] += (i2 - 1);
                    p_mod[2] += (i3 - 1);

                    auto d = distance_naive.box_distance(p_mod, box);
                    dist_naive = std::min(d, dist_naive);
                }
            }
        }

        ASSERT_FLOAT_EQ(dist, dist_naive);
    }
}
