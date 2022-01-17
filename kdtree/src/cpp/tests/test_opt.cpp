#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <kdtree/kdtree.hpp>
#include <kdtree/kdtree_opt.hpp>
#include <kdtree/kdtree_utils.hpp>
#include <kdtree/tournament_tree.hpp>

using namespace wenda::kdtree;

typedef PriorityQueue<std::pair<float, uint32_t>, PairLessFirst> PairPriorityQueue;
typedef TournamentTree<std::pair<float, uint32_t>, PairLessFirst> PairTournamentTree;

class TournamentTreeFixture : public ::testing::TestWithParam<int> {};

TEST_P(TournamentTreeFixture, TournamentTreeTest) {
    int num_max = GetParam();

    std::vector<int> values(1000);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    std::generate(values.begin(), values.end(), [&]() { return dist(rng); });

    wenda::kdtree::TournamentTree<int> tree(num_max, std::numeric_limits<int>::max());

    for (auto const &value : values) {
        if (value < tree.top()) {
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
    TournamentTreePriority, TournamentTreeFixture,
    testing::Values(2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 23));
