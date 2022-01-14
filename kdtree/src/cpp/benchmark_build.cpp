#include <benchmark/benchmark.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include "kdtree.hpp"
#include "kdtree_impl.hpp"
#include "kdtree_build_opt.hpp"
#include "kdtree_utils.hpp"

namespace kdt = wenda::kdtree;

namespace {

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
