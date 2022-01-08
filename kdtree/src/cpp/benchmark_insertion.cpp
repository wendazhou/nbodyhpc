#include <array>
#include <vector>

#include <benchmark/benchmark.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include "kdtree.hpp"
#include "kdtree_opt.hpp"
#include "tournament_tree.hpp"

namespace kdt = wenda::kdtree;

namespace {

std::vector<kdt::PositionAndIndex> fill_random_positions_and_index(int n, unsigned int seed) {
    std::vector<kdt::PositionAndIndex> positions(n);

    typedef r123::Philox4x32 RNG;
    RNG rng;

    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = seed;

    for (int i = 0; i < n; ++i) {
        c.v[0] = i;
        auto r = rng(c, uk);

        positions[i].position[0] = r123::u01<float>(r[0]);
        positions[i].position[1] = r123::u01<float>(r[1]);
        positions[i].position[2] = r123::u01<float>(r[2]);
        positions[i].index = i;
    }

    return positions;
}

/** This template contains a policy which always operates on the same subset of positions,
 * making it possible for those to be held in fast cache.
 * 
 */
template <typename Inserter> struct Cached {
    tcb::span<const kdt::PositionAndIndex> positions_;
    Inserter &inserter_;
    typename Inserter::distance_t distance_;
    uint32_t query_size_;

    Cached(
        Inserter &inserter, typename Inserter::distance_t distance,
        tcb::span<const kdt::PositionAndIndex> positions, uint32_t query_size, unsigned int)
        : positions_(positions), inserter_(inserter), distance_(distance), query_size_(query_size) {
    }

    float operator()(uint32_t index, std::array<float, 3> const &query) {
        typename Inserter::queue_t queue(32, {std::numeric_limits<float>::max(), -1});
        inserter_(positions_.subspan(0, query_size_), query, queue, distance_);
        return queue.top().first;
    }
};

/** This template contains a policy which operates on contiguous positions throughout the array,
 * predictably advancing through the positions array. This represents a scenario which is
 * favourable to the prefetcher, but still requires memory bandwidth.
 * 
 */
template <typename Inserter> struct Contiguous {
    tcb::span<const kdt::PositionAndIndex> positions_;
    Inserter &inserter_;
    typename Inserter::distance_t distance_;
    uint32_t query_size_;

    Contiguous(
        Inserter &inserter, typename Inserter::distance_t distance,
        tcb::span<const kdt::PositionAndIndex> positions, uint32_t query_size, unsigned int)
        : positions_(positions), inserter_(inserter), distance_(distance), query_size_(query_size) {
    }

    float operator()(uint32_t index, std::array<float, 3> const &query) {
        typename Inserter::queue_t queue(32, {std::numeric_limits<float>::max(), -1});
        auto offset = (index * query_size_) % positions_.size();

        inserter_(positions_.subspan(offset, query_size_), query, queue, distance_);
        return queue.top().first;
    }
};

/** This template contains a policy which operates on random blocks throughout the array,
 * preventing large-scale prefetching of the data. However, it still operates contiguously
 * on blocks of size `query_size`, which makes it somewhat more optimistic than the real-world
 * use.
 * 
 */
template <typename Inserter> struct RandomBlock {
    typedef r123::Philox4x32 RNG;

    RNG::ukey_type uk_ = {{}};
    RNG rng_;

    tcb::span<const kdt::PositionAndIndex> positions_;
    Inserter &inserter_;
    typename Inserter::distance_t distance_;
    uint32_t query_size_;

    RandomBlock(
        Inserter &inserter, typename Inserter::distance_t distance,
        tcb::span<const kdt::PositionAndIndex> positions, uint32_t query_size, unsigned int seed)
        : positions_(positions), inserter_(inserter), distance_(distance), query_size_(query_size) {
        uk_[0] = seed;
    }

    float operator()(uint32_t index, std::array<float, 3> const &query) {
        RNG::ctr_type c = {{}};
        c[0] = index;

        typename Inserter::queue_t queue(32, {std::numeric_limits<float>::max(), -1});

        auto r = rng_(c, uk_);
        auto offset = r[0] % (positions_.size() - query_size_);

        inserter_(positions_.subspan(offset, query_size_), query, queue, distance_);

        return queue.top().first;
    }
};

template <
    template <typename, typename> typename Inserter, template <typename, typename> typename Queue>
using InserterL2 = Inserter<
    wenda::kdtree::L2Distance, Queue<std::pair<float, uint32_t>, wenda::kdtree::PairLessFirst>>;

template <
    template <typename, typename> typename InserterT, template <typename, typename> typename QueueT,
    template <typename> typename Loop>
void Insertion(benchmark::State &state) {
    size_t num_points = state.range(0);
    size_t query_size = state.range(1);

    auto positions = fill_random_positions_and_index(num_points, 42);
    auto positions_span = tcb::make_span(positions);
    std::array<float, 3> query = {0.5, 0.5, 0.5};

    typedef InserterL2<InserterT, QueueT> Inserter;
    Inserter inserter;
    typename Inserter::distance_t distance;

    Loop<Inserter> loop(inserter, distance, positions_span, query_size, 43);

    uint32_t idx = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(loop(idx, query));
        idx += 1;
    }

    state.SetBytesProcessed(state.iterations() * query_size * sizeof(kdt::PositionAndIndex));
}

void Memcpy(benchmark::State &state) {
    size_t num_points = state.range(0);
    size_t query_size = state.range(1);

    auto positions = fill_random_positions_and_index(num_points, 42);
    auto positions_span = tcb::make_span(positions);

    std::vector<kdt::PositionAndIndex> positions_copy(query_size);

    uint32_t idx = 0;

    for (auto _ : state) {
        benchmark::ClobberMemory();
        std::copy(positions_span.begin(), positions_span.begin() + query_size,
            positions_copy.begin());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * query_size * sizeof(kdt::PositionAndIndex));
}

} // namespace

#define DEFINE_BENCHMARKS_ALL_INSERTERS(Queue, Loop) \
    BENCHMARK_TEMPLATE(Insertion, kdt::InsertShorterDistanceVanilla, Queue, Loop)->Args({1000000, 1024}); \
    BENCHMARK_TEMPLATE(Insertion, kdt::InsertShorterDistanceUnrolled4, Queue, Loop)->Args({1000000, 1024}); \
    BENCHMARK_TEMPLATE(Insertion, kdt::InsertShorterDistanceAVX, Queue, Loop)->Args({1000000, 1024});

DEFINE_BENCHMARKS_ALL_INSERTERS(kdt::TournamentTree, RandomBlock)
DEFINE_BENCHMARKS_ALL_INSERTERS(kdt::TournamentTree, Contiguous)
DEFINE_BENCHMARKS_ALL_INSERTERS(kdt::TournamentTree, Cached)

#undef DEFINE_BENCHMARKS_ALL_INSERTERS

BENCHMARK(Memcpy)->Args({1000000, 1024});
