#include <benchmark/benchmark.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include "kdtree.hpp"
#include "kdtree_build_opt.hpp"
#include "kdtree_impl.hpp"
#include "kdtree_utils.hpp"
#include "floyd_rivest.hpp"

namespace kdt = wenda::kdtree;

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

template <typename SelectionPolicy = kdt::detail::FloydRivestSelectionPolicy>
void build_tree(std::vector<kdt::PositionAndIndex> &positions) {
    std::vector<kdt::KDTree::KDTreeNode> nodes;

    kdt::detail::KDTreeBuilder<kdt::detail::NullSynchonization, SelectionPolicy> builder(
        nodes, positions, 32, 8);
    builder.build(0);
}

template <typename SelectionPolicy> void benchmark_build_tree(benchmark::State &state) {
    uint32_t i = 0;

    for (auto _ : state) {
        state.PauseTiming();
        auto positions = kdt::make_random_position_and_index(state.range(0), i);
        state.ResumeTiming();

        build_tree<SelectionPolicy>(positions);
        ++i;
    }
}

void benchmark_avx2_select(benchmark::State &state) {
    uint32_t i = 0;

    uint32_t num_points = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> data(num_points);
        fill_random(data.begin(), data.end(), i, [](auto r) { return r123::u01<float>(r[0]); });
        state.ResumeTiming();

        kdt::detail::quickselect_float_array(data.data(), data.size(), data.size() / 2);

        benchmark::DoNotOptimize(data[data.size() / 2]);
    }
}

void benchmark_stdlib_select(benchmark::State &state) {
    uint32_t i = 0;

    uint32_t num_points = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> data(num_points);
        fill_random(data.begin(), data.end(), i, [](auto r) { return r123::u01<float>(r[0]); });
        state.ResumeTiming();

        std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());

        benchmark::DoNotOptimize(data[data.size() / 2]);
    }
}

void benchmark_floyd_rivest_select(benchmark::State &state) {
    uint32_t i = 0;

    uint32_t num_points = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> data(num_points);
        fill_random(data.begin(), data.end(), i, [](auto r) { return r123::u01<float>(r[0]); });
        state.ResumeTiming();

        wenda::kdtree::floyd_rivest_select(data.begin(), data.begin() + data.size() / 2, data.end(), std::less<float>{});

        benchmark::DoNotOptimize(data[data.size() / 2]);
    }
}

} // namespace

BENCHMARK_TEMPLATE(benchmark_build_tree, kdt::detail::FloydRivestSelectionPolicy)
    ->Arg(1 << 20)
    ->Arg(1 << 22)
    ->Arg(1 << 24)
    ->Unit(benchmark::kSecond);

BENCHMARK_TEMPLATE(benchmark_build_tree, kdt::detail::CxxSelectionPolicy)
    ->Arg(1 << 20)
    ->Arg(1 << 22)
    ->Arg(1 << 24)
    ->Unit(benchmark::kSecond);

BENCHMARK(benchmark_avx2_select)->Range(256, 2 << 15);
BENCHMARK(benchmark_stdlib_select)->Range(256, 2 << 15);
BENCHMARK(benchmark_floyd_rivest_select)->Range(256, 2 << 15);
