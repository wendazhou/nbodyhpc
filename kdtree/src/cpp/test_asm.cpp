#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "kdtree_opt.hpp"
#include "kdtree_utils.hpp"
#include "tournament_tree.hpp"

extern "C" {
void tournament_tree_update_root(
    void *tree, uint32_t idx, float element_value, uint32_t element_idx);

void tournament_tree_replace_top(void *tree, float element_value, uint32_t element_idx);

void wenda_insert_closest_l2_avx2(
    float const *const *positions, size_t n, float const *query, void *tree,
    uint32_t const *indices);
void wenda_insert_closest_l2_periodic_avx2(
    float const *const *positions, size_t n, float const *query, void *tree,
    uint32_t const *indices, float boxsize);
}

namespace {

typedef wenda::kdtree::TournamentTree<std::pair<float, uint32_t>, wenda::kdtree::PairLessFirst>
    PairTournamentTree;

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

    PairTournamentTree tree(tree_size, {std::numeric_limits<float>::max(), -1});
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

    PairTournamentTree tree(tree_size, {std::numeric_limits<float>::max(), -1});
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
    ::testing::Combine(::testing::Values(1, 4, 13), ::testing::Values(1, 2, 3, 4, 8, 16, 27, 100)));

//! Parametrized test fixture for inserters
//! The first parameter represents the number of neighbors to obtain,
//! The second parameter represents the number of points to generate.
class InserterAsmTest : public ::testing::TestWithParam<std::tuple<int, int>> {
  public:
    std::vector<uint32_t> seeds_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
};

template <typename DistanceT, typename Fn, typename SeedContainer>
void insert_l2_test(
    int num_closest, int num_positions, DistanceT const &distance, SeedContainer const &seeds,
    Fn const &fn) {
    for (int seed : seeds) {
        auto positions = wenda::kdtree::make_random_position_and_index(num_positions, seed);
        auto tree = PairTournamentTree(num_closest, {std::numeric_limits<float>::max(), -1});
        auto tree2 = PairTournamentTree(tree);

        std::array<float, 3> query = {0.4f, 0.5f, 0.6f};

        fn(positions, query, tree, distance);

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

TEST_P(InserterAsmTest, InsertL2) {
    int num_closest;
    int num_positions;

    std::tie(num_closest, num_positions) = GetParam();
    wenda::kdtree::L2Distance distance;

    insert_l2_test(
        num_closest,
        num_positions,
        distance,
        seeds_,
        [](auto const &positions, auto const &query, auto &tree, auto const &distance) {
            auto positions_array = wenda::kdtree::PositionAndIndexArray(positions);
            wenda_insert_closest_l2_avx2(
                positions_array.positions_.data(),
                positions_array.size(),
                query.data(),
                tree.data().data(),
                positions_array.indices_.data());
        });
}

TEST_P(InserterAsmTest, InsertL2Periodic) {
    int num_closest;
    int num_positions;

    std::tie(num_closest, num_positions) = GetParam();
    wenda::kdtree::L2PeriodicDistance<float> distance{1.1f};

    insert_l2_test(
        num_closest,
        num_positions,
        distance,
        seeds_,
        [](auto const &positions, auto const &query, auto &tree, auto const &distance) {
            auto positions_array = wenda::kdtree::PositionAndIndexArray(positions);

            wenda_insert_closest_l2_periodic_avx2(
                positions_array.positions_.data(),
                positions_array.size(),
                query.data(),
                tree.data().data(),
                positions_array.indices_.data(),
                distance.box_size_);
        });
}

INSTANTIATE_TEST_SUITE_P(
    InserterAsm, InserterAsmTest,
    ::testing::Combine(::testing::Values(1, 3, 7, 8, 16, 20), ::testing::Values(8, 16, 32, 256)));
