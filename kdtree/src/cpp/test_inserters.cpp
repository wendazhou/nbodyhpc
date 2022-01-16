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

template <typename Inserter, typename DistanceT, typename ContainerT>
std::vector<std::pair<float, uint32_t>> find_nearest_inserter(
    ContainerT const &positions, std::array<float, 3> const &query, int k,
    DistanceT const &distance) {

    typedef kdt::TournamentTree<std::pair<float, uint32_t>, kdt::PairLessFirst> QueueT;

    QueueT queue(k, {std::numeric_limits<float>::max(), -1});

    Inserter inserter;
    auto positions_soa = kdt::PositionAndIndexArray(positions);

    inserter(positions_soa, query, queue, distance);

    std::vector<std::pair<float, uint32_t>> result(k);
    queue.copy_values(result.begin());
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

    typedef kdt::L2Distance distance_t;

    const distance_t distance_;
};

template <template <typename, typename> typename Inserter> struct InserterL2PeriodicHolder {
    typedef std::pair<float, uint32_t> result_t;
    typedef kdt::detail::KDTreeQuery<
        kdt::L2PeriodicDistance<float>, kdt::TournamentTree<result_t, kdt::PairLessFirst>, Inserter>
        query_t;
    typedef Inserter<
        kdt::L2PeriodicDistance<float>, kdt::TournamentTree<result_t, kdt::PairLessFirst>>
        inserter_t;

    typedef kdt::L2PeriodicDistance<float> distance_t;
    const distance_t distance_{2.0f};
};

using Inserters = ::testing::Types<
    InserterL2Holder<kdt::InsertShorterDistanceVanilla>,
    InserterL2Holder<kdt::InsertShorterDistanceUnrolled4>,
    InserterL2Holder<kdt::InsertShorterDistanceAVX>,
    InserterL2Holder<kdt::InsertShorterDistanceAsm>,
    InserterL2PeriodicHolder<kdt::InsertShorterDistanceVanilla>,
    InserterL2PeriodicHolder<kdt::InsertShorterDistanceUnrolled4>,
    InserterL2PeriodicHolder<kdt::InsertShorterDistanceAVX>,
    InserterL2PeriodicHolder<kdt::InsertShorterDistanceAsm>>;

struct InserterTestConfig {
    unsigned int seed;
    unsigned int num_neighbors;
    unsigned int num_points;
};

} // namespace

template <typename InserterHolder> class KDTreeRandomTestInserterL2 : public ::testing::Test {
  public:
    typedef std::pair<float, uint32_t> result_t;
    typedef typename InserterHolder::query_t query_t;
    typedef typename InserterHolder::inserter_t inserter_t;
    typedef typename InserterHolder::distance_t distance_t;
    distance_t distance_{InserterHolder{}.distance_};

    // Dummy pointers to access the defined type aliases in test functions through decltype
    query_t *dummy_query_ptr_ = nullptr;
    result_t *dummy_result_ptr_ = nullptr;
    inserter_t *dummy_inserter_ptr_ = nullptr;

    std::vector<InserterTestConfig> configs_ = {
        {42, 1, 64}, {43, 4, 128}, {44, 7, 128}, {45, 13, 136}, {46, 17, 256}};
};

TYPED_TEST_SUITE_P(KDTreeRandomTestInserterL2);

TYPED_TEST_P(KDTreeRandomTestInserterL2, BuildAndFindNearest) {
    typedef std::remove_pointer_t<decltype(this->dummy_query_ptr_)> query_t;
    typedef std::remove_pointer_t<decltype(this->dummy_result_ptr_)> result_t;

    for (auto const &config : this->configs_) {
        auto positions = wenda::kdtree::make_random_position_and_index(
            4 * config.num_points + config.seed, config.seed);
        std::array<float, 3> query_pos = {0.4, 0.5, 0.6};

        auto tree = wenda::kdtree::KDTree(std::vector(positions), {.leaf_size = 64});
        query_t query{tree, this->distance_, query_pos, config.num_neighbors};
        query.compute(tree.nodes().data());

        std::vector<result_t> result(config.num_neighbors);
        query.distances_.copy_values(result.data());
        std::sort(result.begin(), result.end());
        for (auto &p : result) {
            p.first = this->distance_.postprocess(p.first);
        }

        auto naive_result =
            find_nearest_naive(positions, query_pos, config.num_neighbors, this->distance_);

        EXPECT_EQ(result, naive_result)
            << "seed: " << config.seed << " num_points: " << config.num_points
            << " num_neighbors: " << config.num_neighbors;
    }
}

TYPED_TEST_P(KDTreeRandomTestInserterL2, FindNearestFlat) {
    typedef std::remove_pointer_t<decltype(this->dummy_inserter_ptr_)> inserter_t;

    for (auto const &config : this->configs_) {
        auto positions =
            wenda::kdtree::make_random_position_and_index(config.num_points, config.seed);
        auto queries = wenda::kdtree::make_random_position_and_index(4, 2 * config.seed);

        for (auto const &query_and_idx : queries) {
            auto const &query = query_and_idx.position;
            auto result_naive =
                find_nearest_naive(positions, query, config.num_neighbors, this->distance_);
            auto result = find_nearest_inserter<inserter_t>(
                positions, query, config.num_neighbors, this->distance_);

            EXPECT_EQ(result_naive, result)
                << "seed: " << config.seed << ", num_points: " << config.num_points
                << ", num_neighbors: " << config.num_neighbors;
        }
    }
}

TYPED_TEST_P(KDTreeRandomTestInserterL2, ExhaustiveLeaves) {
    typedef std::remove_pointer_t<decltype(this->dummy_result_ptr_)> result_t;
    typedef std::remove_pointer_t<decltype(this->dummy_inserter_ptr_)> inserter_t;

    auto positions = wenda::kdtree::make_random_position_and_index(64, 42);
    std::array<float, 3> query_pos = {0.4, 0.5, 0.6};

    auto tree = wenda::kdtree::KDTree(std::vector(positions), {.leaf_size = 16});

    for (int num_points : {1, 4, 7}) {
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

            EXPECT_EQ(result, result_naive)
                << "Error with num_points: " << num_points << " at node: (" << node.left_ << ", "
                << node.right_ << ")";
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(
    KDTreeRandomTestInserterL2, BuildAndFindNearest, ExhaustiveLeaves, FindNearestFlat);
INSTANTIATE_TYPED_TEST_SUITE_P(TestInserter, KDTreeRandomTestInserterL2, Inserters);
