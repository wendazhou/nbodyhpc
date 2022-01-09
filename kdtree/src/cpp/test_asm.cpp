#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "kdtree_utils.h"
#include "tournament_tree.hpp"

extern "C" {
void tournament_tree_update_root(
    void *tree, uint32_t idx, float element_value, uint32_t element_idx);

uint32_t wenda_find_closest_l2_avx2(void *positions, size_t n, float *const query);
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
    auto winner_value = data_[0].first;
    data_[insert_idx] = {element, insert_idx};
    tournament_tree_update_root(data_.data(), insert_idx, element.first, element.second);
}

} // namespace

class TournamentTreeAsmRandomTest : public ::testing::TestWithParam<std::pair<int, int>> {};

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

INSTANTIATE_TEST_SUITE_P(
    TournamentTreeAsm, TournamentTreeAsmRandomTest,
    ::testing::Values(
        std::pair<int, int>{4, 1}, std::pair<int, int>{4, 2}, std::pair<int, int>{4, 4},
        std::pair<int, int>{4, 8}, std::pair<int, int>{13, 1}, std::pair<int, int>{13, 5},
        std::pair<int, int>{13, 27}));

TEST(InserterAsm, InsertFindClosest) {
    wenda::kdtree::L2Distance distance;
    auto positions = wenda::kdtree::make_random_position_and_index(128, 42);
    std::array<float, 3> query = {0.5f, 0.5f, 0.5f};

    auto closest_idx_asm =
        wenda_find_closest_l2_avx2(positions.data(), positions.size(), query.data());

    auto result = std::transform_reduce(
        positions.begin(), positions.end(),
        std::pair<float, uint32_t>{std::numeric_limits<float>::max(), -1},
        [](auto const& p1, auto const& p2) {
            return p1.first < p2.first ? p1 : p2;
        },
        [&](wenda::kdtree::PositionAndIndex const& p) {
            return std::make_pair(distance(p.position, query), p.index);
        });

    ASSERT_EQ(result.second, closest_idx_asm);
}

TEST(InserterAsm, InsertFindClosestMultiple) {
    wenda::kdtree::L2Distance distance;
    auto positions = wenda::kdtree::make_random_position_and_index(256, 42);
    std::array<float, 3> query = {0.5f, 0.5f, 0.5f};

    auto closest_idx_asm_1 =
        wenda_find_closest_l2_avx2(positions.data(), 128, query.data());

    auto closest_idx_asm_2 =
        wenda_find_closest_l2_avx2(positions.data() + 128, 128, query.data());

    auto result = std::transform_reduce(
        positions.begin(), positions.end(),
        std::pair<float, uint32_t>{std::numeric_limits<float>::max(), -1},
        [](auto const& p1, auto const& p2) {
            return p1.first < p2.first ? p1 : p2;
        },
        [&](wenda::kdtree::PositionAndIndex const& p) {
            return std::make_pair(distance(p.position, query), p.index);
        });

    if (result.second < 128) {
        ASSERT_EQ(result.second, closest_idx_asm_1);
    }
    else {
        ASSERT_EQ(result.second - 128, closest_idx_asm_2);
    }
}
