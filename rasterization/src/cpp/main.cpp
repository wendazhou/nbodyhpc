#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <optional>
#include <string_view>
#include <tuple>

#include "lodepng.h"
#include "point_renderer.h"
#include "vertex_utilities.h"
#include "vulkan_support.h"

void save_png_grayscale(tcb::span<float> data, uint32_t grid_size, std::string const &filename) {
    auto [it_min, it_max] = std::minmax_element(data.begin(), data.end());
    float data_min = *it_min;
    float data_max = *it_max;

    std::vector<unsigned char> image(grid_size * grid_size);
    for (uint32_t i = 0; i < grid_size * grid_size; ++i) {
        image[i] =
            static_cast<unsigned char>((data[i] - data_min) / (data_max - data_min) * 255.0f);
    }

    unsigned error = lodepng::encode(filename, image, grid_size, grid_size, LCT_GREY, 8);
    if (error) {
        std::cerr << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}

std::vector<float>
render_vertices(float box_size, uint32_t grid_size, size_t num_slices, std::vector<wenda::vulkan::Vertex> vertices) {
    // Create Vulkan instance, context + device
    wenda::vulkan::VulkanContainer vulkan(false);
    wenda::vulkan::PointRenderer renderer(vulkan, {.width = grid_size, .height = grid_size, .subsample_factor = 4});

    std::vector<float> result(static_cast<size_t>(grid_size) * grid_size * num_slices);

    std::cout << "Start rendering volume" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    renderer.render_points_volume(vertices, box_size, num_slices, result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "Done rendering volume in " << duration.count() << " seconds" << std::endl;

    return result;
}

void render_single_sphere(size_t grid_size, size_t num_slices) {
    // prepare vertex data
    std::vector<wenda::vulkan::Vertex> vertices = {
        {{0.5f, 0.5f, 0.5f}, {1.0f}, 0.25f},
    };

    // Create Vulkan instance, context + device
    auto result = render_vertices(1.0f, static_cast<uint32_t>(grid_size), num_slices, vertices);

    auto total_sum = std::reduce(result.begin(), result.end(), 0.0, std::plus<double>());
    std::cout << "Total weight: " << total_sum << std::endl;

    auto total_pixels = std::transform_reduce(
        result.begin(), result.end(), 0u, std::plus<>{}, [](float v) { return v > 0.0f ? 1ul : 0ul; });
    std::cout << "Proportion pixels: " << static_cast<double>(total_pixels) / (grid_size * grid_size * grid_size) << std::endl;

    auto central_pixel = result[grid_size * grid_size * grid_size / 2 + grid_size * grid_size / 2 + grid_size / 2];
    std::cout << "Central pixel: " << central_pixel << std::endl;

    save_png_grayscale({result.data() + (grid_size / 2) * grid_size * grid_size, grid_size * grid_size}, grid_size, "sphere.png");

    if (grid_size < 64) {
        std::stringstream ss;
        ss << "sphere_" << grid_size << ".bin";
        auto filename = ss.str();

        std::cout << "Saving result to file " << filename << std::endl;
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<char const*>(result.data()), result.size() * sizeof(float));
        file.close();
    }
}

void render_points_from_file(uint32_t grid_size, size_t num_slices, const char *filepath) {
    std::ifstream file(filepath, std::ios::binary);

    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();

    std::uint32_t num_points = file_size / sizeof(wenda::vulkan::Vertex);
    file.seekg(0, std::ios::beg);

    std::cout << "Reading " << num_points << " points from file" << std::endl;

    std::vector<wenda::vulkan::Vertex> vertices(num_points);

    file.read(
        reinterpret_cast<char *>(vertices.data()), sizeof(wenda::vulkan::Vertex) * num_points);

    auto total_radius = std::transform_reduce(
        vertices.begin(),
        vertices.end(),
        0.0f,
        std::plus<>(),
        [](wenda::vulkan::Vertex const &v) { return v.radius; });

    auto input_mass = std::transform_reduce(
        vertices.begin(),
        vertices.end(),
        0.0,
        std::plus<>(),
        [](wenda::vulkan::Vertex const &v) { return v.weight; });

    std::cout << "Average radius: " << total_radius / num_points << std::endl;

    float box_size = 25.0f;
    wenda::augment_vertices_periodic(vertices, {box_size, box_size, box_size});
    wenda::sort_vertices(vertices);

    std::cout << "Vertices after augmentation: " << vertices.size() << std::endl;

    auto result = render_vertices(box_size, grid_size, num_slices, vertices);

    // need higher precision for the total sum
    auto rendered_mass = std::reduce(result.begin(), result.end(), 0.0);

    std::cout << "Mass initial: " << input_mass << std::endl;
    std::cout << "Mass rendered: " << rendered_mass << std::endl;
    std::cout << "Mass reconstruction: " << rendered_mass / input_mass << std::endl;

    std::vector<float> result_out(grid_size * grid_size);

    auto slice_begin = result.begin() + (grid_size / 2) * grid_size * grid_size;
    auto slice_end = slice_begin + grid_size * grid_size;

    std::transform(slice_begin, slice_end, result_out.begin(), std::log1pf);

    save_png_grayscale(result_out, grid_size, "points.png");
}

int main(int argc, char *argv[]) {
    uint32_t grid_size = 512;
    size_t num_slices = 512;

    if (argc > 1) {
        grid_size = std::stoi(argv[1]);
    }

    if (argc > 2) {
        std::cout << "Rendering points from file." << std::endl;
        render_points_from_file(grid_size, num_slices, argv[2]);
    } else {
        render_single_sphere(grid_size, num_slices);
    }

    return 0;
}
