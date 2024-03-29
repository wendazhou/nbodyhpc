#include <array>
#include <execution>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include <kdtree/kdtree.hpp>
#include <kdtree/kdtree_opt.hpp>
#include <kdtree/kdtree_opt_asm.hpp>
#include <kdtree/kdtree_utils.hpp>
#include <kdtree/tournament_tree.hpp>

namespace kdt = wenda::kdtree;

namespace {

/** This template contains a policy which always operates on the same subset of positions,
 * making it possible for those to be held in fast cache.
 *
 */
template <typename Inserter, typename ContainerT> struct Cached {
    ContainerT const &positions_;
    Inserter &inserter_;
    typename Inserter::distance_t distance_;
    uint32_t query_size_;

    Cached(
        Inserter &inserter, typename Inserter::distance_t distance, ContainerT const &positions,
        uint32_t query_size, unsigned int)
        : positions_(positions), inserter_(inserter), distance_(distance), query_size_(query_size) {
    }

    float operator()(uint32_t index, std::array<float, 3> const &query) {
        typename Inserter::queue_t queue(32, {std::numeric_limits<float>::max(), -1});
        inserter_(
            kdt::OffsetRangeContainerWrapper{positions_, 0, query_size_}, query, queue, distance_);
        return queue.top().first;
    }
};

/** This template contains a policy which operates on contiguous positions throughout the array,
 * predictably advancing through the positions array. This represents a scenario which is
 * favourable to the prefetcher, but still requires memory bandwidth.
 *
 */
template <typename Inserter, typename ContainerT> struct Contiguous {
    ContainerT const &positions_;
    Inserter &inserter_;
    typename Inserter::distance_t distance_;
    uint32_t query_size_;

    Contiguous(
        Inserter &inserter, typename Inserter::distance_t distance, ContainerT const &positions,
        uint32_t query_size, unsigned int)
        : positions_(positions), inserter_(inserter), distance_(distance), query_size_(query_size) {
    }

    float operator()(uint32_t index, std::array<float, 3> const &query) {
        typename Inserter::queue_t queue(32, {std::numeric_limits<float>::max(), -1});
        auto offset = (index * query_size_) % (positions_.size() - query_size_);
        offset = (offset / 16) * 16;

        inserter_(
            kdt::OffsetRangeContainerWrapper{positions_, offset, query_size_},
            query,
            queue,
            distance_);
        return queue.top().first;
    }
};

/** This template contains a policy which operates on random blocks throughout the array,
 * preventing large-scale prefetching of the data. However, it still operates contiguously
 * on blocks of size `query_size`, which makes it somewhat more optimistic than the real-world
 * use.
 *
 */
template <typename Inserter, typename ContainerT> struct RandomBlock {
    typedef r123::Philox4x32 RNG;

    RNG::ukey_type uk_ = {{}};
    RNG rng_;

    ContainerT const &positions_;
    Inserter &inserter_;
    typename Inserter::distance_t distance_;
    uint32_t query_size_;

    RandomBlock(
        Inserter &inserter, typename Inserter::distance_t distance, ContainerT const &positions,
        uint32_t query_size, unsigned int seed)
        : positions_(positions), inserter_(inserter), distance_(distance), query_size_(query_size) {
        uk_[0] = seed;
    }

    float operator()(uint32_t index, std::array<float, 3> const &query) {
        RNG::ctr_type c = {{}};
        c[0] = index;

        typename Inserter::queue_t queue(32, {std::numeric_limits<float>::max(), -1});

        auto r = rng_(c, uk_);
        auto offset = r[0] % (positions_.size() - query_size_);
        offset = (offset / 16) * 16;

        inserter_(
            kdt::OffsetRangeContainerWrapper{positions_, offset, query_size_},
            query,
            queue,
            distance_);

        return queue.top().first;
    }
};

template <
    template <typename, typename> typename Inserter, template <typename, typename> typename Queue>
using InserterL2 = Inserter<
    wenda::kdtree::L2Distance, Queue<std::pair<float, uint32_t>, wenda::kdtree::PairLessFirst>>;

/** Benchmark the given brute-force insertion method, using the given queue type.
 *
 */
template <
    template <typename, typename> typename InserterT, template <typename, typename> typename QueueT,
    template <typename, typename> typename Loop>
void Insertion(benchmark::State &state) {
    size_t num_points = state.range(0);
    size_t query_size = state.range(1);

    // force 32-element alignment
    num_points = (num_points / 32) * 32;

    auto positions = kdt::make_random_position_and_index_array(num_points, 42);
    std::array<float, 3> query = {0.4, 0.5, 0.6};

    typedef InserterL2<InserterT, QueueT> Inserter;
    Inserter inserter;
    typename Inserter::distance_t distance;

    Loop<Inserter, decltype(positions)> loop(inserter, distance, positions, query_size, 43);

    uint32_t idx = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(loop(idx, query));
        idx += 1;
    }

    state.SetBytesProcessed(state.iterations() * query_size * sizeof(decltype(positions)::value_type));
}

template <
    template <typename, typename> typename InserterT, template <typename, typename> typename QueueT>
using InserterL2Periodic = InserterT<
    wenda::kdtree::L2PeriodicDistance<float>,
    QueueT<std::pair<float, uint32_t>, wenda::kdtree::PairLessFirst>>;

template <
    template <typename, typename> typename InserterT, template <typename, typename> typename QueueT,
    template <typename, typename> typename Loop>
void InsertionPeriodic(benchmark::State &state) {
    size_t num_points = state.range(0);
    size_t query_size = state.range(1);

    // force 32-element alignment
    num_points = (num_points / 32) * 32;

    auto positions = kdt::make_random_position_and_index_array(num_points, 42);
    std::array<float, 3> query = {0.4, 0.5, 0.6};

    typedef InserterL2Periodic<InserterT, QueueT> Inserter;
    Inserter inserter;
    typename Inserter::distance_t distance{1.0f};

    Loop<Inserter, decltype(positions)> loop(inserter, distance, positions, query_size, 43);

    uint32_t idx = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(loop(idx, query));
        idx += 1;
    }

    state.SetBytesProcessed(state.iterations() * query_size * sizeof(decltype(positions)::value_type));
}

void Memcpy(benchmark::State &state) {
    size_t num_points = state.range(0);
    size_t query_size = state.range(1);

    auto positions = kdt::make_random_position_and_index(num_points, 42);
    auto positions_span = tcb::make_span(positions);

    decltype(positions) positions_copy(query_size);

    size_t idx = 0;

    for (auto _ : state) {
        std::copy(
            positions_span.begin() + idx * query_size,
            positions_span.begin() + idx * query_size + query_size,
            positions_copy.begin());
        benchmark::DoNotOptimize(positions_copy.data());
        benchmark::ClobberMemory();
        idx = (idx + 1) % (num_points / query_size);
    }

    state.SetBytesProcessed(state.iterations() * query_size * sizeof(positions[0]));
}

void ReduceDistance(benchmark::State &state) {
    size_t num_points = state.range(0);
    size_t query_size = state.range(1);

    auto positions = kdt::make_random_position_and_index(num_points, 42);
    auto positions_span = tcb::make_span(positions);

    std::array<float, 3> query = {0.5, 0.5, 0.5};
    kdt::L2Distance distance;

    size_t idx = 0;

    for (auto _ : state) {
        auto subspan = positions_span.subspan(idx * query_size, query_size);

        float result = std::transform_reduce(
#if __cpp_lib_execution >= 201902
            std::execution::unseq,
#elif defined(__cpp_lib_execution)
            std::execution::seq,
#endif
            subspan.begin(),
            subspan.end(),
            std::numeric_limits<float>::max(),
            [](float a, float b) { return std::min(a, b); },
            [&](auto const &p) { return distance(p.position, query); });
        benchmark::DoNotOptimize(result);

        idx = (idx + 1) % (num_points / query_size);
    }

    state.SetBytesProcessed(state.iterations() * query_size * sizeof(kdt::PositionAndIndex<3>));
}


} // namespace

#define DEFINE_BENCHMARKS_ALL_INSERTERS(Procedure, Queue, Loop)                                    \
    BENCHMARK_TEMPLATE(Procedure, kdt::InsertShorterDistanceVanilla, Queue, Loop)                  \
        ->Args({1000000, 1024});                                                                   \
    BENCHMARK_TEMPLATE(Procedure, kdt::InsertShorterDistanceUnrolled4, Queue, Loop)                \
        ->Args({1000000, 1024});                                                                   \
    BENCHMARK_TEMPLATE(Procedure, kdt::InsertShorterDistanceAVX, Queue, Loop)                      \
        ->Args({1000000, 1024});                                                                   \
    BENCHMARK_TEMPLATE(Procedure, kdt::InsertShorterDistanceAsm, Queue, Loop)                      \
        ->Args({1000000, 1024});

DEFINE_BENCHMARKS_ALL_INSERTERS(Insertion, kdt::TournamentTree, RandomBlock)
// DEFINE_BENCHMARKS_ALL_INSERTERS(Insertion, kdt::TournamentTree, Contiguous)
DEFINE_BENCHMARKS_ALL_INSERTERS(Insertion, kdt::TournamentTree, Cached)

DEFINE_BENCHMARKS_ALL_INSERTERS(InsertionPeriodic, kdt::TournamentTree, RandomBlock)
// DEFINE_BENCHMARKS_ALL_INSERTERS(InsertionPeriodic, kdt::TournamentTree, Contiguous)
DEFINE_BENCHMARKS_ALL_INSERTERS(InsertionPeriodic, kdt::TournamentTree, Cached)

#undef DEFINE_BENCHMARKS_ALL_INSERTERS

BENCHMARK(ReduceDistance)->Args({1000000, 1024});
BENCHMARK(Memcpy)->Args({1000000, 1024});
