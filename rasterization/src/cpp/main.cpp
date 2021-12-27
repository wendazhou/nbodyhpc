#include <array>
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
    uint32_t grid_size = 400;

    // prepare vertex data
    std::vector<wenda::vulkan::Vertex> vertices = {
        {{0.5f, 0.5f, 0.0f}, {1.0f}, 0.25f},
    };

    // Create Vulkan instance, context + device
    wenda::vulkan::VulkanContainer vulkan;

    wenda::vulkan::PointRenderer renderer(vulkan, 1.0f, grid_size);

    std::vector<uint8_t> result (grid_size * grid_size * grid_size * sizeof(float));
    renderer.render_points_volume(vertices, {reinterpret_cast<float*>(result.data()), grid_size * grid_size * grid_size});

    lodepng::encode("out.png", result, 800, 800);

    return 0;
}
