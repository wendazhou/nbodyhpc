#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "kdtree_utils.hpp"

namespace {

template <typename ContainerT, typename Distance>
std::vector<std::pair<float, uint32_t>> find_nearest_naive(
    ContainerT const &positions, const std::array<float, 3> &query, int k,
    Distance const &distance) {
    std::priority_queue<std::pair<float, uint32_t>> distances(
        {}, std::vector<std::pair<float, uint32_t>>(k, {std::numeric_limits<float>::max(), -1}));

    for (auto const &pos_and_index : positions) {
        auto dist = distance(pos_and_index.position, query);
        if (dist < distances.top().first) {
            distances.pop();
            distances.push({dist, pos_and_index.index});
        }
    }

    std::vector result(std::move(wenda::kdtree::get_container_from_adapter(distances)));
    std::sort(result.begin(), result.end());

    for (auto &p : result) {
        p.first = distance.postprocess(p.first);
    }

    return result;
}

} // namespace

class KDTreeRandomTest : public ::testing::TestWithParam<int> {};

TEST_P(KDTreeRandomTest, BuildAndFindNearestClass) {
    int block_size = 8;
    std::array<float, 3> query = {0.4, 0.5, 0.6};

    auto positions =
        wenda::kdtree::make_random_position_and_index_array(GetParam(), 42, 1.0, block_size);

    auto tree = wenda::kdtree::KDTree(positions, {.leaf_size = 8, .block_size = block_size});

    ASSERT_TRUE(
        std::all_of(tree.nodes().begin(), tree.nodes().end(), [block_size](auto const &node) {
            return node.dimension_ != -1 || ((node.right_ - node.left_) % block_size == 0);
        }));

    wenda::kdtree::KDTreeQueryStatistics statistics;
    auto result = tree.find_closest(query, 4, wenda::kdtree::L2Distance{}, &statistics);

    auto naive_result = find_nearest_naive(positions, query, 4, wenda::kdtree::L2Distance{});

    ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));

    ASSERT_EQ(result, naive_result);
}

TEST_P(KDTreeRandomTest, BuildAndFindNearestClassMT) {
    wenda::kdtree::KDTreeConfiguration tree_config{};

    auto positions = wenda::kdtree::make_random_position_and_index_array(GetParam(), 42, 1.0, tree_config.block_size);
    std::array<float, 3> query = {0.5, 0.5, 0.5};

    auto tree = wenda::kdtree::KDTree(positions, tree_config);

    ASSERT_TRUE(std::all_of(tree.nodes().begin(), tree.nodes().end(), [](auto const &node) {
        return node.dimension_ != -1 || ((node.right_ - node.left_) % 8 == 0);
    }));

    wenda::kdtree::KDTreeQueryStatistics statistics;
    auto result = tree.find_closest(query, 4, wenda::kdtree::L2Distance{}, &statistics);

    auto naive_result = find_nearest_naive(positions, query, 4, wenda::kdtree::L2Distance{});

    ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));

    ASSERT_EQ(result, naive_result);
}

TEST_P(KDTreeRandomTest, BuildAndFindNearestPeriodic) {
    wenda::kdtree::KDTreeConfiguration tree_config{};
    float boxsize = 2.0f;

    auto positions = wenda::kdtree::make_random_position_and_index_array(GetParam(), 42, boxsize, tree_config.block_size);
    auto queries = wenda::kdtree::make_random_position_and_index(100, 43, boxsize);

    auto distance = wenda::kdtree::L2PeriodicDistance<float>{boxsize};

    auto tree = wenda::kdtree::KDTree(positions, tree_config);
    wenda::kdtree::KDTreeQueryStatistics statistics;

    for (auto &query_and_idx : queries) {
        auto &query = query_and_idx.position;
        auto result = tree.find_closest(query, 4, distance, &statistics);

        auto naive_result = find_nearest_naive(positions, query, 4, distance);

        ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));

        ASSERT_EQ(result, naive_result);
    }
}

INSTANTIATE_TEST_SUITE_P(
    BuildAndFindNearestSmall, KDTreeRandomTest, testing::Values(10, 100, 1000));

TEST(KDTreeMetric, TestL2PeriodicBox3D) {
    auto positions = wenda::kdtree::make_random_position_and_index(100, 42);
    std::array<float, 6> box = {0.2f, 0.5f, 0.4f, 0.6f, 0.0f, 0.1f};

    wenda::kdtree::L2Distance distance_naive;
    wenda::kdtree::L2PeriodicDistance<float> distance_periodic{1.0f};

    for (auto const &pos_and_index : positions) {
        auto const &p = pos_and_index.position;
        auto dist = distance_periodic.box_distance(p, box);

        auto dist_naive = std::numeric_limits<float>::max();

        for (int i1 = 0; i1 < 3; ++i1) {
            for (int i2 = 0; i2 < 3; ++i2) {
                for (int i3 = 0; i3 < 3; ++i3) {
                    auto p_mod = p;
                    p_mod[0] += (i1 - 1);
                    p_mod[1] += (i2 - 1);
                    p_mod[2] += (i3 - 1);

                    auto d = distance_naive.box_distance(p_mod, box);
                    dist_naive = std::min(d, dist_naive);
                }
            }
        }

        ASSERT_NEAR(dist, dist_naive, 1e-6);
    }
}
