#include <kdtree/kdtree.hpp>

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

#include <kdtree/kdtree_build_opt.hpp>
#include <kdtree/kdtree_impl.hpp>
#include <kdtree/kdtree_opt.hpp>
#include <kdtree/kdtree_opt_asm.hpp>
#include <kdtree/kdtree_utils.hpp>
#include <kdtree/tournament_tree.hpp>
#include <span.hpp>

namespace {
std::vector<wenda::kdtree::PositionAndIndex<3>> resize_to_following_multiple(
    std::vector<wenda::kdtree::PositionAndIndex<3>> &&positions, size_t block_size) {

    auto size_up = (positions.size() + block_size - 1) / block_size * block_size;
    positions.resize(
        size_up,
        wenda::kdtree::PositionAndIndex<3>{
            {std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()},
            -1u});

    return std::move(positions);
}
} // namespace

namespace wenda {

namespace kdtree {

#ifdef _MSC_VER
// MSVC does not support C++ standard aligned alloc
void *aligned_alloc(size_t alignment, size_t size) noexcept {
    return _aligned_malloc(size, alignment);
}

void aligned_free(void *ptr) noexcept { _aligned_free(ptr); }
#else
void *aligned_alloc(size_t alignment, size_t size) noexcept {
    // round up size to next multiple of alignment
    size = (size + alignment - 1) & ~(alignment - 1);
    // For some reason std::aligned_alloc not exposed in libstdc++ on MacOS
    return ::aligned_alloc(alignment, size);
}

void aligned_free(void *ptr) noexcept { return free(ptr); }
#endif

PositionAndIndexArray<3, float, uint32_t>
make_position_and_indices(tcb::span<const std::array<float, 3>> const &positions, int block_size) {
    size_t size_up;

    if (block_size <= 0) {
        size_up = positions.size();
    } else {
        size_up = (positions.size() + block_size - 1) / block_size * block_size;
    }

    PositionAndIndexArray<3, float, uint32_t> result(size_up);
    std::iota(result.indices_.begin(), result.indices_.end(), 0);

    for (size_t dim = 0; dim < 3; ++dim) {
        std::transform(
            positions.begin(), positions.end(), result.positions_[dim], [dim](auto const &pos) {
                return pos[dim];
            });

        std::fill(
            result.positions_[dim] + positions.size(),
            result.positions_[dim] + size_up,
            std::numeric_limits<float>::max());
    }

    return result;
}

KDTree::KDTree(tcb::span<const std::array<float, 3>> positions, KDTreeConfiguration const &config)
    : KDTree(make_position_and_indices(positions, config.block_size), config) {}

KDTree::KDTree(PositionAndIndexArray<3> positions, KDTreeConfiguration const &config)
    : positions_(std::move(positions)), config_(config) {

    if (positions_.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("More than uint32_t points are not supported.");
    }

    if (config.block_size % 8 != 0) {
        throw std::runtime_error("block_size must be a multiple of 8.");
    }

    if (positions_.size() % config.block_size != 0) {
        throw std::runtime_error("block_size must divide the number of points.");
    }

    const size_t num_points_per_thread = 5000000;
    int max_threads =
        config.max_threads == -1 ? std::thread::hardware_concurrency() : config.max_threads;

    typedef detail::FloydRivestAvxSelectionPolicy SelectionPolicy;

    if (false && max_threads > 1 && positions_.size() >= 2 * num_points_per_thread) {
        double num_threads = static_cast<double>(positions_.size()) / num_points_per_thread;
        num_threads = std::max(num_threads, static_cast<double>(max_threads));

        int log2_threads = static_cast<int>(std::log2(num_threads));

        typedef detail::MutexLockSynchronization Synchronization;
        detail::KDTreeBuilder<Synchronization, SelectionPolicy> builder(
            nodes_, positions_, config.leaf_size, config.block_size);
        builder.build(log2_threads);
    } else {
        detail::KDTreeBuilder<detail::NullSynchonization, SelectionPolicy> builder(
            nodes_, positions_, config.leaf_size, config.block_size);
        builder.build(0);
    }
}

template <typename Distance>
std::vector<std::pair<float, uint32_t>> KDTree::find_closest(
    std::array<float, 3> const &position, size_t k, Distance const &distance,
    KDTreeQueryStatistics *statistics) const {

    typedef std::pair<float, uint32_t> result_t;
    detail::KDTreeQuery<Distance, TournamentTree<result_t, PairLessFirst>, InsertShorterDistanceAsm>
        query(nodes_, positions_, distance, position, k);
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
