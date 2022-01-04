#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <cxxopts.h>

#include "kdtree.hpp"

namespace {

template <typename Dist>
std::vector<std::array<float, 3>> fill_random_positions(int n, Dist const &dist) {
    std::vector<std::array<float, 3>> positions(n);

    for (int i = 0; i < n; ++i) {
        positions[i][0] = dist();
        positions[i][1] = dist();
        positions[i][2] = dist();
    }

    return positions;
}

struct BenchmarkResult {
    std::chrono::duration<double> build_time;
    std::chrono::duration<double> query_time;
    double points_visited_proportion;
    size_t num_queries;
};

struct BenchmarkConfig {
    size_t num_queries;
    int num_neighbors = 16;
};

BenchmarkResult benchmark_lookup_same(
    tcb::span<const std::array<float, 3>> positions, BenchmarkConfig const &config,
    wenda::kdtree::KDTreeConfiguration const &tree_config = {}) {
    std::chrono::high_resolution_clock clock;

    auto tree_build_start_t = clock.now();
    auto tree = wenda::kdtree::KDTree(positions, tree_config);
    auto tree_build_end_t = clock.now();

    float total_distance = 0;
    size_t total_points_visited = 0;

    wenda::kdtree::KDTreeQueryStatistics statistics;
    auto num_queries = std::min(config.num_queries, positions.size());

    auto tree_query_start_t = clock.now();

    for (size_t i = 0; i < num_queries; ++i) {
        auto const &pos = positions[i];
        auto nearest = tree.find_closest(pos, config.num_neighbors, &statistics);
        total_distance += nearest[0];
        total_points_visited += statistics.points_visited;
    }

    auto tree_query_end_t = clock.now();

    if (total_distance != 0) {
        std::cout << "Total distance was not 0!" << std::endl;
    }

    return {
        tree_build_end_t - tree_build_start_t,
        tree_query_end_t - tree_query_start_t,
        static_cast<double>(total_points_visited) / (positions.size() * num_queries),
        num_queries};
}

BenchmarkResult benchmark_kdtree_random(
    int n, BenchmarkConfig const &config,
    wenda::kdtree::KDTreeConfiguration const &tree_config = {}) {

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto positions = fill_random_positions(n, [&]() { return dist(rng); });
    return benchmark_lookup_same(positions, config, tree_config);
}

std::vector<std::array<float, 3>> read_array_from_file(std::string const &filepath) {
    std::ifstream file(filepath, std::ios::in | std::ios::binary | std::ios::ate);
    auto file_size = file.tellg();

    std::uint32_t num_points = static_cast<uint32_t>(file_size / (sizeof(float) * 3));
    file.seekg(0, std::ios::beg);

    std::vector<std::array<float, 3>> positions(num_points);
    file.read(reinterpret_cast<char *>(positions.data()), positions.size() * sizeof(float) * 3);

    return positions;
}

BenchmarkResult benchmark_kdtree_file(
    std::string const &filepath, BenchmarkConfig const &config,
    wenda::kdtree::KDTreeConfiguration const &tree_config = {}) {
    auto positions = read_array_from_file(filepath);
    return benchmark_lookup_same(positions, config, tree_config);
}

} // namespace

int main(int argc, char *argv[]) {
    cxxopts::Options options("kdtree", "KD-tree benchmark");

    // clang-format off
    options.add_options()
        ("n,num-points", "Number of points to use", cxxopts::value<uint32_t>()->default_value("10000000"))
        ("num-neighbors", "Number of neighbors to find", cxxopts::value<int>()->default_value("16"))
        ("q,num-queries", "Number of queries to perform", cxxopts::value<uint32_t>()->default_value("500000"))
        ("t,threads", "Number of threads to use", cxxopts::value<int>()->default_value("-1"))
        ("leaf-size", "Size of kd-tree leaves", cxxopts::value<int>()->default_value("8"))
        ("f,file", "File to use for benchmarking", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print help");
    // clang-format on

    auto opts = options.parse(argc, argv);

    BenchmarkResult result;

    auto filepath = opts["file"].as<std::string>();

    BenchmarkConfig config {
        .num_queries = opts["num-queries"].as<uint32_t>(),
        .num_neighbors = opts["num-neighbors"].as<int>()
    };

    wenda::kdtree::KDTreeConfiguration tree_config {
        .leaf_size = opts["leaf-size"].as<int>(),
        .max_threads = opts["threads"].as<int>()
    };

    if (filepath.size() == 0) {
        auto num_points = opts["num-points"].as<uint32_t>();
        std::cout << "Benchmarking kdtree with " << num_points << " points" << std::endl;
        result = benchmark_kdtree_random(num_points, config, tree_config);
    } else {
        std::string filepath = argv[1];
        std::cout << "Benchmarking kdtree with data from: " << filepath << std::endl;
        result = benchmark_kdtree_file(filepath, config, tree_config);
    }

    std::cout << "Build time: " << result.build_time.count() << "s" << std::endl;
    std::cout << "Query time: " << result.query_time.count() << "s" << std::endl;
    std::cout << "Query performance: " << result.num_queries / result.query_time.count() << " qps"
              << std::endl;
    std::cout << "Points visited proportion: " << result.points_visited_proportion * 100 << "%"
              << std::endl;
}
