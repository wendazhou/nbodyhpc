#include "floyd_rivest.hpp"

#include <algorithm>
#include <vector>

#include <Random123/philox.h>
#include <gtest/gtest.h>

#include "kdtree_build_opt.hpp"
#include "kdtree_impl.hpp"
#include "kdtree_utils.hpp"

namespace {

template <typename It> void fill_random_integers(It beg, It end, uint32_t seed) {
    typedef r123::Philox4x32 RNG;
    RNG rng;

    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};

    uk[0] = seed;

    uint32_t idx = 0;

    while (beg != end) {
        c.v[0] = idx;
        auto r = rng(c, uk);
        *beg = r[0];
        ++beg;
    }
}

struct LessInDimension {
    int dimension_;

    template <typename T> bool operator()(T const &lhs, T const &rhs) const {
        return lhs.position[dimension_] < rhs.position[dimension_];
    }
};

} // namespace

TEST(FloydRivestTest, TestFloydRivestMedianInt) {
    std::vector<int> values(10000);
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
