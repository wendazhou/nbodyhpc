#include "kdtree.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <vector>

#include "kdtree_impl.hpp"
#include "kdtree_opt.hpp"
#include "kdtree_opt_asm.hpp"
#include "kdtree_utils.hpp"
#include "tournament_tree.hpp"
#include <span.hpp>

namespace wenda {

namespace kdtree {

std::vector<PositionAndIndex>
make_position_and_indices(tcb::span<const std::array<float, 3>> const &positions) {
    std::vector<PositionAndIndex> result(positions.size());

    for (size_t i = 0; i < positions.size(); ++i) {
        result[i].position = positions[i];
        result[i].index = i;
    }

    return result;
}

KDTree::KDTree(tcb::span<const std::array<float, 3>> positions, KDTreeConfiguration const &config)
    : KDTree(make_position_and_indices(positions), config) {}

KDTree::KDTree(
    std::vector<PositionAndIndex> &&positions_and_indices, KDTreeConfiguration const &config)
    : positions_() {

    if (positions_and_indices.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("More than uint32_t points are not supported.");
    }

    const size_t num_points_per_thread = 5000000;
    int max_threads =
        config.max_threads == -1 ? std::thread::hardware_concurrency() : config.max_threads;

    auto size_up = (positions_and_indices.size() + config.block_size - 1) / config.block_size *
                   config.block_size;
    positions_and_indices.resize(
        size_up,
        PositionAndIndex{
            {std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()},
            -1u});

    positions_ = PositionAndIndexArray(std::move(positions_and_indices));

    if (max_threads > 1 && positions_and_indices.size() >= 2 * num_points_per_thread) {
        double num_threads = static_cast<double>(positions_.size()) / num_points_per_thread;
        num_threads = std::max(num_threads, static_cast<double>(max_threads));

        int log2_threads = static_cast<int>(std::log2(num_threads));

        typedef detail::MutexLockSynchronization Synchronization;
        detail::KDTreeBuilder<Synchronization> builder(
            nodes_, positions_, config.leaf_size, config.block_size);
        builder.build(log2_threads);
    } else {
        detail::KDTreeBuilder<detail::NullSynchonization> builder(
            nodes_, positions_, config.leaf_size, config.block_size);
        builder.build(0);
    }
}

template <typename Distance>
std::vector<std::pair<float, uint32_t>> KDTree::find_closest(
    std::array<float, 3> const &position, size_t k, Distance const &distance,
    KDTreeQueryStatistics *statistics) const {

    typedef std::pair<float, uint32_t> result_t;

    detail::
        KDTreeQuery<Distance, TournamentTree<result_t, PairLessFirst>, InsertShorterDistanceAsm>
            query(*this, distance, position, k);
    query.compute(&nodes_[0]);

    if (statistics) {
        statistics->nodes_visited = query.num_nodes_visited;
        statistics->nodes_pruned = query.num_nodes_pruned;
        statistics->points_visited = query.num_points_visited;
    }

    std::vector<std::pair<float, uint32_t>> result(k);
    query.distances_.copy_values(result.begin());
    std::sort(result.begin(), result.end(), PairLessFirst{});

    // convert back to distances from distance-squared
    for (auto &p : result) {
        p.first = distance.postprocess(p.first);
    }

    return result;
}

template std::vector<std::pair<float, uint32_t>> KDTree::find_closest<L2Distance>(
    std::array<float, 3> const &position, size_t k, L2Distance const &,
    KDTreeQueryStatistics *statistics) const;

template std::vector<std::pair<float, uint32_t>> KDTree::find_closest<L2PeriodicDistance<float>>(
    std::array<float, 3> const &position, size_t k, L2PeriodicDistance<float> const &,
    KDTreeQueryStatistics *statistics) const;

} // namespace kdtree

} // namespace wenda
