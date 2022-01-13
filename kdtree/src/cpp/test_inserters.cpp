#include <queue>
#include <type_traits>

#include <gtest/gtest.h>

#include "kdtree.hpp"
#include "kdtree_impl.hpp"
#include "kdtree_opt.hpp"
#include "kdtree_opt_asm.hpp"
#include "kdtree_utils.hpp"

namespace kdt = wenda::kdtree;

namespace {

template <typename Container, typename Distance>
std::vector<std::pair<float, uint32_t>> find_nearest_naive(
    Container const &positions, const std::array<float, 3> &query, int k,
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

template <template <typename, typename> typename Inserter> struct InserterL2Holder {
    typedef std::pair<float, uint32_t> result_t;
    typedef kdt::detail::KDTreeQuery<
        kdt::L2Distance, kdt::TournamentTree<result_t, kdt::PairLessFirst>, Inserter>
        query_t;
    typedef Inserter<kdt::L2Distance, kdt::TournamentTree<result_t, kdt::PairLessFirst>> inserter_t;
};

using InsertersL2 = ::testing::Types<
    InserterL2Holder<kdt::InsertShorterDistanceVanilla>,
    InserterL2Holder<kdt::InsertShorterDistanceUnrolled4>,
    InserterL2Holder<kdt::InsertShorterDistanceAVX>,
    InserterL2Holder<kdt::InsertShorterDistanceAsm>>;

template <template <typename, typename> typename Inserter> struct InserterL2PeriodicHolder {
    typedef std::pair<float, uint32_t> result_t;
    typedef kdt::detail::KDTreeQuery<
        kdt::L2PeriodicDistance<float>, kdt::TournamentTree<result_t, kdt::PairLessFirst>, Inserter>
        query_t;
};

using InsertersL2Periodic = ::testing::Types<
    InserterL2PeriodicHolder<kdt::InsertShorterDistanceVanilla>,
    InserterL2PeriodicHolder<kdt::InsertShorterDistanceUnrolled4>,
    InserterL2PeriodicHolder<kdt::InsertShorterDistanceAVX>>;

} // namespace

template <typename InserterHolder> class KDTreeRandomTestInserterL2 : public ::testing::Test {
  public:
    typedef std::pair<float, uint32_t> result_t;
    typedef typename InserterHolder::query_t query_t;
    typedef typename InserterHolder::inserter_t inserter_t;
    kdt::L2Distance distance_;

    // Dummy pointers to access the defined type aliases in test functions through decltype
    query_t *dummy_query_ptr_ = nullptr;
    result_t *dummy_result_ptr_ = nullptr;
    inserter_t *dummy_inserter_ptr_ = nullptr;
};

TYPED_TEST_SUITE_P(KDTreeRandomTestInserterL2);

TYPED_TEST_P(KDTreeRandomTestInserterL2, BuildAndFindNearest) {
    typedef std::remove_pointer_t<decltype(this->dummy_query_ptr_)> query_t;
    typedef std::remove_pointer_t<decltype(this->dummy_result_ptr_)> result_t;

    uint32_t num_points = 1;

    auto positions = wenda::kdtree::make_random_position_and_index(10000, 42);
    std::array<float, 3> query_pos = {0.4, 0.5, 0.6};

    auto tree = wenda::kdtree::KDTree(std::vector(positions), {.leaf_size = 64});
    query_t query{tree, this->distance_, query_pos, num_points};
    query.compute(tree.nodes().data());

    std::vector<result_t> result(num_points);
    query.distances_.copy_values(result.data());
    std::sort(result.begin(), result.end());
    for (auto &p : result) {
        p.first = this->distance_.postprocess(p.first);
    }

    auto naive_result = find_nearest_naive(positions, query_pos, num_points, this->distance_);

    ASSERT_EQ(result, naive_result);
}

TYPED_TEST_P(KDTreeRandomTestInserterL2, ExhaustiveLeaves) {
    typedef std::remove_pointer_t<decltype(this->dummy_result_ptr_)> result_t;
    typedef std::remove_pointer_t<decltype(this->dummy_inserter_ptr_)> inserter_t;

    uint32_t num_points = 1;

    auto positions = wenda::kdtree::make_random_position_and_index(256, 42);
    std::array<float, 3> query_pos = {0.4, 0.5, 0.6};

    auto tree = wenda::kdtree::KDTree(std::vector(positions), {.leaf_size = 64});

    for (auto const &node : tree.nodes()) {
        if (node.dimension_ != -1) {
            continue;
        }

        kdt::TournamentTree<result_t, kdt::PairLessFirst> tournament_tree(
            num_points, {std::numeric_limits<float>::max(), -1});
        auto span = wenda::kdtree::OffsetRangeContainerWrapper(
            tree.positions(), node.left_, node.right_ - node.left_);

        inserter_t inserter;
        inserter(span, query_pos, tournament_tree, {});

        std::vector<result_t> result(num_points);
        tournament_tree.copy_values(result.data());
        std::sort(result.begin(), result.end());
        for (auto &p : result) {
            p.first = this->distance_.postprocess(p.first);
        }

        auto result_naive = find_nearest_naive(span, query_pos, num_points, this->distance_);

        ASSERT_EQ(result, result_naive);
    }
}

REGISTER_TYPED_TEST_SUITE_P(KDTreeRandomTestInserterL2, BuildAndFindNearest, ExhaustiveLeaves);
INSTANTIATE_TYPED_TEST_SUITE_P(TestInserterL2, KDTreeRandomTestInserterL2, InsertersL2);

template <typename InserterHolder>
class KDTreeRandomTestInserterL2Periodic : public ::testing::Test {
  public:
    typedef std::pair<float, uint32_t> result_t;
    typedef typename InserterHolder::query_t query_t;
    kdt::L2PeriodicDistance<float> distance_ = {2.0f};

    // Dummy pointers to access the defined type aliases in test functions through decltype
    query_t *dummy_query_ptr_ = nullptr;
    result_t *dummy_result_ptr_ = nullptr;

    std::vector<int> seeds_ = {42, 43, 44, 45, 46, 47, 48, 49};
};
TYPED_TEST_SUITE_P(KDTreeRandomTestInserterL2Periodic);

TYPED_TEST_P(KDTreeRandomTestInserterL2Periodic, BuildAndFindNearest) {
    typedef std::remove_pointer_t<decltype(this->dummy_query_ptr_)> query_t;
    typedef std::remove_pointer_t<decltype(this->dummy_result_ptr_)> result_t;

    uint32_t num_points = 17;

    for (auto seed : this->seeds_) {
        auto positions =
            wenda::kdtree::make_random_position_and_index(100, 42, this->distance_.box_size_);
        std::array<float, 3> query_pos = {0.4, 0.5, 0.6};

        auto tree = wenda::kdtree::KDTree(std::vector(positions), {.leaf_size = 64});
        query_t query{tree, this->distance_, query_pos, num_points};
        query.compute(tree.nodes().data());

        std::vector<result_t> result(num_points);
        query.distances_.copy_values(result.data());
        std::sort(result.begin(), result.end());
        for (auto &p : result) {
            p.first = this->distance_.postprocess(p.first);
        }

        auto naive_result = find_nearest_naive(positions, query_pos, num_points, this->distance_);

        ASSERT_EQ(result, naive_result);
    }
}
REGISTER_TYPED_TEST_SUITE_P(KDTreeRandomTestInserterL2Periodic, BuildAndFindNearest);
INSTANTIATE_TYPED_TEST_SUITE_P(
    TestInserterL2Periodic, KDTreeRandomTestInserterL2Periodic, InsertersL2Periodic);
