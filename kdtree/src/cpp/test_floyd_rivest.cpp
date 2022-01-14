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

struct LessInDimension {
    int dimension_;

    template <typename T> bool operator()(T const &lhs, T const &rhs) const {
        return lhs.position[dimension_] < rhs.position[dimension_];
    }
};

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
        positions.begin(), median_it, positions.end(), LessInDimension{0});

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
    for (uint32_t i = 0; i < 100; ++i) {
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
    for (uint32_t i = 0; i < 100; ++i) {
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

template <typename SelectionPolicy> class TestSelectionPolicy : public ::testing::Test {
  public:
    SelectionPolicy selection_;
};

} // namespace

TYPED_TEST_SUITE_P(TestSelectionPolicy);

TYPED_TEST_P(TestSelectionPolicy, SelectMedian) {
    auto positions = wenda::kdtree::make_random_position_and_index_array(10000, 42);
    int dimension = 0;

    auto median_offset = positions.size() / 2;
    auto median_it = positions.begin() + median_offset;
    this->selection_(positions.begin(), median_it, positions.end(), dimension);

    auto median_value = (*median_it).position[dimension];

    std::nth_element(
        positions.positions_[dimension],
        positions.positions_[dimension] + median_offset,
        positions.positions_[0] + positions.size());

    auto expected_median_value = positions.positions_[dimension][median_offset];

    ASSERT_EQ(median_value, expected_median_value);
}

REGISTER_TYPED_TEST_SUITE_P(TestSelectionPolicy, SelectMedian);

typedef ::testing::Types<
    wenda::kdtree::detail::FloydRivestSelectionPolicy,
    wenda::kdtree::detail::FloydRivestOptSelectionPolicy>
    SelectionPolicies;
INSTANTIATE_TYPED_TEST_SUITE_P(TestSelection, TestSelectionPolicy, SelectionPolicies);
