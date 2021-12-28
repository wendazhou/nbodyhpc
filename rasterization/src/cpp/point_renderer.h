#pragma once

#include <memory>
#include <functional>
#include <vector>

#include "vertex_utilities.h"
#include "vulkan_support.h"
#include "span.hpp"

namespace wenda {
namespace vulkan {

namespace util {
    bool always_false() noexcept;
}

class PointRendererImpl;

using Vertex = wenda::Vertex;

struct PointRendererConfiguration {
    uint32_t grid_size;
    uint32_t subsample_factor = 4;
};

class PointRenderer {
    std::unique_ptr<PointRendererImpl> impl_;
    VulkanContainer const& container_;

    uint32_t grid_size_;
public:
    PointRenderer(VulkanContainer const& container, PointRendererConfiguration const& config);
    ~PointRenderer();

    PointRenderer(PointRenderer&&) noexcept = default;

    void render_points(tcb::span<const Vertex> points, float box_size, tcb::span<float> result);
    void render_points_volume(tcb::span<const Vertex> points, float box_size, tcb::span<float> result, std::function<bool()> const& should_stop = util::always_false);

    uint32_t grid_size() const noexcept { return grid_size_; }
};

} // namespace vulkan
} // namespace wenda
