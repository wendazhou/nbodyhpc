#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "kdtree_utils.hpp"
#include "tournament_tree.hpp"

extern "C" {
void tournament_tree_update_root(
    void *tree, uint32_t idx, float element_value, uint32_t element_idx);

void tournament_tree_replace_top(void *tree, float element_value, uint32_t element_idx);
uint32_t wenda_find_closest_l2_avx2(void *positions, size_t n, float const *query);
void wenda_insert_closest_l2_avx2(void *positions, size_t n, float const *query, void *tree);
}

namespace {

struct PairLessFirst {
    template <typename T> bool operator()(T const &a, T const &b) {
        return std::get<0>(a) < std::get<0>(b);
    }
};

class PairTournamentTree
    : public wenda::kdtree::TournamentTree<std::pair<float, uint32_t>, PairLessFirst> {
  public:
    PairTournamentTree(int n)
        : TournamentTree<std::pair<float, uint32_t>, PairLessFirst>(
              n, {std::numeric_limits<float>::max(), -1}) {}

    std::vector<std::pair<std::pair<float, uint32_t>, uint32_t>> &data() { return data_; }
};

void update_tournament_tree_asm(
    PairTournamentTree &tree, std::pair<float, uint32_t> const &element) {
    auto &data_ = tree.data();
    auto insert_idx = data_[0].second;
    data_[insert_idx] = {element, insert_idx};
    tournament_tree_update_root(data_.data(), insert_idx, element.first, element.second);
}

} // namespace

class TournamentTreeAsmRandomTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(TournamentTreeAsmRandomTest, TestUpdateFromRoot) {
    int tree_size;
    int num_updates;

    std::tie(tree_size, num_updates) = GetParam();

    PairTournamentTree tree(tree_size);
    PairTournamentTree tree2(tree);

    std::vector<std::pair<float, uint32_t>> updates(num_updates);
    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < num_updates; ++i) {
        updates[i] = {dist(rng), i};
    }

    for (auto const &u : updates) {
        tree.replace_top(u);
        update_tournament_tree_asm(tree2, u);
        ASSERT_EQ(tree.data(), tree2.data());
    }
}

TEST_P(TournamentTreeAsmRandomTest, TestReplaceTop) {
    int tree_size;
    int num_updates;

    std::tie(tree_size, num_updates) = GetParam();

    PairTournamentTree tree(tree_size);
    PairTournamentTree tree2(tree);

    std::vector<std::pair<float, uint32_t>> updates(num_updates);
    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < num_updates; ++i) {
        updates[i] = {dist(rng), i};
    }

    for (auto const &u : updates) {
        tree.replace_top(u);
        tournament_tree_replace_top(tree2.data().data(), u.first, u.second);
        ASSERT_EQ(tree.data(), tree2.data());
    }
}

INSTANTIATE_TEST_SUITE_P(
    TournamentTreeAsm, TournamentTreeAsmRandomTest,
    ::testing::Combine(
        ::testing::Values(1, 4, 13),
        ::testing::Values(1, 2, 3, 4, 8, 16, 27, 100)
    ));

class FindClosestAsmTest : public ::testing::TestWithParam<int> {
  public:
    std::vector<uint32_t> seeds_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
};

TEST_P(FindClosestAsmTest, FindClosest) {
    wenda::kdtree::L2Distance distance;
    auto length = GetParam();

    for (int seed : seeds_) {
        auto positions = wenda::kdtree::make_random_position_and_index(length, seed);
        std::array<float, 3> query = {0.5f, 0.5f, 0.5f};

        auto closest_idx_asm =
            wenda_find_closest_l2_avx2(positions.data(), positions.size(), query.data());

        auto result = std::transform_reduce(
            positions.begin(),
            positions.end(),
            std::pair<float, uint32_t>{std::numeric_limits<float>::max(), -1},
            [](auto const &p1, auto const &p2) { return p1.first < p2.first ? p1 : p2; },
            [&](wenda::kdtree::PositionAndIndex const &p) {
                return std::make_pair(distance(p.position, query), p.index);
            });

        ASSERT_EQ(result.second, closest_idx_asm);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FindClosestAsm, FindClosestAsmTest,
    ::testing::Values(1, 2, 3, 5, 7, 8, 15, 16, 113, 127, 128, 230));

class InserterAsmTest : public ::testing::TestWithParam<std::tuple<int, int>> {
  public:
    std::vector<uint32_t> seeds_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
};

TEST_P(InserterAsmTest, InsertL2) {
    int num_closest;
    int num_positions;

    std::tie(num_closest, num_positions) = GetParam();

    wenda::kdtree::L2Distance distance;

    for (int seed : seeds_) {
        auto positions = wenda::kdtree::make_random_position_and_index(num_positions, seed);
        auto tree = PairTournamentTree(num_closest);
        auto tree2 = PairTournamentTree(num_closest);

        std::array<float, 3> query = {0.5f, 0.5f, 0.5f};

        wenda_insert_closest_l2_avx2(
            positions.data(), positions.size(), query.data(), tree.data().data());

        for (auto const &p : positions) {
            auto dist = distance(p.position, query);
            if (dist < tree2.top().first) {
                tree2.replace_top({dist, p.index});
            }
        }

        // we need to copy out to guarantee same results
        // due to arbitrary comparison of equal floats which may
        // differ between algorithms.
        std::vector<std::pair<float, uint32_t>> result(num_closest);
        std::vector<std::pair<float, uint32_t>> result2(num_closest);

        tree.copy_values(result.begin());
        tree2.copy_values(result2.begin());

        std::sort(result.begin(), result.end());
        std::sort(result2.begin(), result2.end());

        // Drop values for which there was no finite value.
        auto it_inf_result = std::find_if(result.begin(), result.end(), [](auto const &p) {
            return p.first == std::numeric_limits<float>::max();
        });
        result.resize(std::distance(result.begin(), it_inf_result));

        auto it_inf_result2 = std::find_if(result2.begin(), result2.end(), [](auto const &p) {
            return p.first == std::numeric_limits<float>::max();
        });
        result2.resize(std::distance(result2.begin(), it_inf_result2));

        ASSERT_EQ(result, result2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    InserterAsm, InserterAsmTest,
    ::testing::Combine(
        ::testing::Values(1, 3, 7, 8, 16, 20),
        ::testing::Values(1, 3, 6, 7, 8, 15, 16, 23, 128, 341)));
