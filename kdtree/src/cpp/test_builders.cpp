#include <vector>

#include "kdtree.hpp"
#include "kdtree_build_opt.hpp"
#include "kdtree_impl.hpp"
#include "kdtree_utils.hpp"

#include <Random123/philox.h>
#include <gtest/gtest.h>

namespace kdt = wenda::kdtree;

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

template <typename Query, typename Distance>
std::vector<std::pair<float, uint32_t>>
extract_result(Query const &query, Distance const &distance) {
    std::vector<std::pair<float, uint32_t>> result(query.distances_.size());
    query.distances_.copy_values(result.begin());
    std::sort(result.begin(), result.end(), kdt::PairLessFirst{});

    // convert back to distances from distance-squared
    for (auto &p : result) {
        p.first = distance.postprocess(p.first);
    }

    return result;
}

template <typename Selection> struct KDTreeBuilderTest : ::testing::Test {
  public:
    Selection *dummy_selection_ptr = nullptr;
    kdt::L2Distance distance_;
};

typedef ::testing::Types<
    kdt::detail::CxxSelectionPolicy, kdt::detail::FloydRivestSelectionPolicy,
    kdt::detail::FloydRivestOptSelectionPolicy, kdt::detail::FloydRivestAvxSelectionPolicy>
    SelectionPolicies;

template <typename NodesRange>
bool check_all_element_covered(NodesRange const &nodes, size_t num_positions) {
    std::vector<int> covered(num_positions, 0);

    for (auto const &node : nodes) {
        if (node.dimension_ != -1) {
            continue;
        }

        for (auto i = node.left_; i < node.right_; ++i) {
            ++covered[i];
        }
    }

    return std::all_of(covered.begin(), covered.end(), [](int c) { return c == 1; });
}

} // namespace

TYPED_TEST_SUITE_P(KDTreeBuilderTest);

TYPED_TEST_P(KDTreeBuilderTest, BuildAndFindNearestClass) {
    auto num_neighbors = 4;

    kdt::KDTreeConfiguration tree_config{.leaf_size = 32};
    auto positions =
        wenda::kdtree::make_random_position_and_index_array(80, 42, 1.0, tree_config.block_size);
    auto positions_copy = kdt::PositionAndIndexArray(positions);

    std::vector<std::tuple<float, float, float>> positions_data(positions.size());
    std::transform(positions.begin(), positions.end(), positions_data.begin(), [](auto const &p) {
        return std::make_tuple(p.position[0], p.position[1], p.position[2]);
    });

    // Build nodes manually
    std::vector<kdt::KDTree::KDTreeNode> nodes;
    typedef std::remove_pointer_t<decltype(this->dummy_selection_ptr)> SelectionPolicy;
    kdt::detail::KDTreeBuilder<kdt::detail::NullSynchonization, SelectionPolicy> builder(
        nodes, positions, tree_config.leaf_size, tree_config.block_size);
    builder.build(0);

    {
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
        EXPECT_EQ(idx_tuple_not_preserved, positions.size())
            << " selection did not preserve tuple at given index.";
    }

    EXPECT_TRUE(check_all_element_covered(nodes, positions.size()));

    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [&](auto const &node) {
        return node.dimension_ != -1 || ((node.right_ - node.left_) % tree_config.block_size == 0);
    }));

    // Query from constructed nodes
    std::array<float, 3> query = {0.4, 0.5, 0.6};
    kdt::detail::KDTreeQuery q(nodes, positions, this->distance_, query, num_neighbors);
    q.compute(&nodes[0]);

    auto result = extract_result(q, this->distance_);
    auto naive_result = find_nearest_naive(positions, query, num_neighbors, this->distance_);

    EXPECT_EQ(result, naive_result)
        << "Results from constructed nodes and naive implementation differ";
}

REGISTER_TYPED_TEST_SUITE_P(KDTreeBuilderTest, BuildAndFindNearestClass);
INSTANTIATE_TYPED_TEST_SUITE_P(TestBuilderPolicies, KDTreeBuilderTest, SelectionPolicies);
