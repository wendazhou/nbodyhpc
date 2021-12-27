#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string_view>
#include <tuple>

#include "lodepng.h"
#include "vulkan_support.h"
#include "point_renderer.h"


int main() {
    uint32_t grid_size = 600;

    // prepare vertex data
    std::vector<wenda::vulkan::Vertex> vertices = {
        {{0.5f, 0.5f, 0.0f}, {1.0f}, 0.25f},
    };

    // Create Vulkan instance, context + device
    wenda::vulkan::VulkanContainer vulkan(true);

    wenda::vulkan::PointRenderer renderer(vulkan, 1.0f, grid_size);

    std::vector<uint8_t> result (grid_size * grid_size * grid_size * sizeof(float));

    for(int i = 0; i < 2; ++i) {
        std::cout << "Rendering frame " << i << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        renderer.render_points_volume(vertices, {reinterpret_cast<float*>(result.data()), grid_size * grid_size * grid_size});
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        std::cout << "Done rendering in: " << duration.count() << " seconds" << std::endl;
    }


    lodepng::encode("out.png", result, grid_size, grid_size);

    return 0;
}
