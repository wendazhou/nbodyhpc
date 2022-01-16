#include "floyd_rivest.hpp"

#include <algorithm>
#include <vector>

#include <Random123/philox.h>
#include <gtest/gtest.h>

#include "kdtree_build_opt.hpp"
#include "kdtree_impl.hpp"
#include "kdtree_utils.hpp"

namespace {

template <typename It, typename Fn> void fill_random(It beg, It end, uint32_t seed, Fn &&fn) {
    typedef r123::Philox4x32 RNG;
    RNG rng;

    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};

    uk[0] = seed;

    uint32_t idx = 0;

    while (beg != end) {
        c.v[0] = idx;
        auto r = rng(c, uk);
        *beg = fn(r);
        ++beg;
        idx += 1;
    }
}

template <typename It> void fill_random_integers(It beg, It end, uint32_t seed) {
    fill_random(beg, end, seed, [](auto r) { return r[0]; });
}

} // namespace

TEST(FloydRivestTest, TestFloydRivestMedianInt) {
    std::vector<int> values(100);
    fill_random_integers(values.begin(), values.end(), 42);

    auto median_it = values.begin() + values.size() / 2;

    wenda::kdtree::floyd_rivest_select(values.begin(), median_it, values.end(), std::less<int>());

    auto fr_median = *median_it;

    std::nth_element(values.begin(), median_it, values.end(), std::less<int>());

    auto cxx_median = *median_it;

    ASSERT_EQ(fr_median, cxx_median);
}

TEST(FloydRivestTest, TestFloydRivestMedianPositionAndIndex) {
    auto positions = wenda::kdtree::make_random_position_and_index_array(10000, 42);

    auto median_it = positions.begin() + positions.size() / 2;

    wenda::kdtree::floyd_rivest_select(
        positions.begin(),
        median_it,
        positions.end(),
        wenda::kdtree::detail::PositionAtDimensionCompare{0});

    auto fr_median_value = (*median_it).position[0];

    std::nth_element(
        positions.positions_[0],
        positions.positions_[0] + positions.size() / 2,
        positions.positions_[0] + positions.size());

    auto cxx_median_value = positions.positions_[0][positions.size() / 2];

    ASSERT_EQ(fr_median_value, cxx_median_value);
}

TEST(SelectionTest, TestFloatArrayPartition) {
    std::vector<float> values(10000);
    fill_random(values.begin(), values.end(), 42, [](auto r) { return r123::u01<float>(r[0]); });

    auto partition_point =
        wenda::kdtree::detail::partition_float_array(values.data(), values.size(), 0.5);

    ASSERT_TRUE(std::all_of(
        values.begin(), values.begin() + partition_point, [](float f) { return f < 0.5; }));
    ASSERT_TRUE(std::all_of(
        values.begin() + partition_point, values.end(), [](float f) { return f >= 0.5; }));
}

TEST(SelectionTest, TestFloatArraySelectionExamples) {
    std::vector<float> values = {0, 1, 0};
    wenda::kdtree::detail::quickselect_float_array(values.data(), values.size(), 1);

    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[1], 0);
    EXPECT_EQ(values[2], 1);
}

TEST(SelectionTest, TestFloatArraySelectionExamples2) {
    std::vector<float> values = {1, 0, 0};
    wenda::kdtree::detail::quickselect_float_array(values.data(), values.size(), 1);

    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[1], 0);
    EXPECT_EQ(values[2], 1);
}

TEST(SelectionTest, TestFloatArraySelection) {
    for (uint32_t i = 0; i < 50; ++i) {
        std::vector<float> values(500 + 73 + i);
        fill_random(values.begin(), values.end(), i, [](auto r) { return r123::u01<float>(r[0]); });

        auto k = values.size() / 2;

        wenda::kdtree::detail::quickselect_float_array(values.data(), values.size(), k);

        auto order_k_statistic = values[k];

        std::nth_element(values.begin(), values.begin() + k, values.end());

        auto cxx_order_k_statistic = values[k];

        EXPECT_EQ(order_k_statistic, cxx_order_k_statistic) << "i = " << i;
    }
}

TEST(SelectionTest, TestFloatArrayFloydRivest) {
    for (uint32_t i = 0; i < 50; ++i) {
        std::vector<float> values(500 + 73 + i);
        fill_random(values.begin(), values.end(), i, [](auto r) { return r123::u01<float>(r[0]); });

        auto k = values.size() / 2;

        wenda::kdtree::detail::floyd_rivest_float_array(values.data(), values.size(), k);

        auto order_k_statistic = values[k];

        std::nth_element(values.begin(), values.begin() + k, values.end());

        auto cxx_order_k_statistic = values[k];

        EXPECT_EQ(order_k_statistic, cxx_order_k_statistic) << "i = " << i;
    }
}

namespace {

struct SelectionTestConfiguration {
    size_t size;
    size_t beg_index;
    size_t mid_index;
    size_t end_index;
};

template <typename SelectionPolicy> class TestSelectionPolicy : public ::testing::Test {
  public:
    SelectionPolicy selection_;

    std::vector<SelectionTestConfiguration> configs_ = {
        {10, 0, 4, 10},
        {10, 2, 6, 7},
        {50, 0, 25, 50},
        {50, 14, 17, 20},
        {1000, 0, 500, 1000},
        {1000, 400, 450, 475},
    };
};

} // namespace

TYPED_TEST_SUITE_P(TestSelectionPolicy);

TYPED_TEST_P(TestSelectionPolicy, SelectMedian) {
    uint32_t i = 0;

    for (auto const &config : this->configs_) {
        auto positions = wenda::kdtree::make_random_position_and_index_array(config.size, 42);
        auto positions_copy = wenda::kdtree::PositionAndIndexArray(positions);

        // make a copy of the positions (without associated indices)
        std::vector<std::tuple<float, float, float>> positions_data(positions.size());
        std::transform(
            positions.begin(), positions.end(), positions_data.begin(), [](auto const &p) {
                return std::make_tuple(p.position[0], p.position[1], p.position[2]);
            });

        int dimension = i % 3;

        auto beg_it = positions.begin() + config.beg_index;
        auto end_it = positions.begin() + config.end_index;

        auto median_it = positions.begin() + config.mid_index;
        this->selection_(beg_it, median_it, end_it, dimension);

        // Check that all tuples have been preserved
        size_t idx_tuple_not_preserved = positions.size();

        for (size_t i = 0; i < positions.size(); ++i) {
            auto pos = positions[i].position;
            auto idx = positions[i].index;
            auto pos_tuple = std::make_tuple(pos[0], pos[1], pos[2]);
            if (positions_data[idx] != pos_tuple) {
                idx_tuple_not_preserved = i;
                break;
            }
        }

        EXPECT_EQ(idx_tuple_not_preserved, positions.size()) << " selection did not preserve tuple at given index.";

        auto median_value = (*median_it).position[dimension];
        auto selection_other_coord = (*median_it).position[(dimension + 1) % 3];
        auto selection_idx = (*median_it).index;

        std::nth_element(
            positions_copy.begin() + config.beg_index,
            positions_copy.begin() + config.mid_index,
            positions_copy.begin() + config.end_index,
            wenda::kdtree::detail::PositionAtDimensionCompare{dimension});

        auto expected_median_value = positions_copy[config.mid_index].position[dimension];
        auto expected_selection_idx = positions_copy[config.mid_index].index;
        auto expected_selection_other_coord =
            positions_copy[config.mid_index].position[(dimension + 1) % 3];

        EXPECT_EQ(median_value, expected_median_value);
        EXPECT_EQ(selection_other_coord, expected_selection_other_coord);
        EXPECT_EQ(selection_idx, expected_selection_idx)
            << "size = " << config.size << ", beg = " << config.beg_index
            << ", mid = " << config.mid_index << ", end = " << config.end_index;
    }
}

REGISTER_TYPED_TEST_SUITE_P(TestSelectionPolicy, SelectMedian);

typedef ::testing::Types<
    wenda::kdtree::detail::CxxSelectionPolicy, wenda::kdtree::detail::FloydRivestSelectionPolicy,
    wenda::kdtree::detail::FloydRivestOptSelectionPolicy,
    wenda::kdtree::detail::FloydRivestAvxSelectionPolicy>
    SelectionPolicies;
INSTANTIATE_TYPED_TEST_SUITE_P(TestSelection, TestSelectionPolicy, SelectionPolicies);
