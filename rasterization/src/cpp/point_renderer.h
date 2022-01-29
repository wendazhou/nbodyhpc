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
    size_t grid_size;
    uint32_t subsample_factor = 4;
};

class PointRenderer {
    std::unique_ptr<PointRendererImpl> impl_;
    VulkanContainer const& container_;

    size_t grid_size_;
public:
    PointRenderer(VulkanContainer const& container, PointRendererConfiguration const& config);
    ~PointRenderer();

    PointRenderer(PointRenderer&&) noexcept = default;

    void render_points(tcb::span<const Vertex> points, float box_size, tcb::span<float> result);

    /** Render points in a volume.
     * 
     * @param points The points to render.
     * @param box_size The size of the volume to render.
     * @param num_slices The number of slices to use in the rendering. If less than grid_size, only the front portion of the volume will be rendered.
     * @param result A span of size at least grid_size ** 3 to store the result.
     * @param should_stop Optional callback to check if the rendering should be interrupted or cancelled.
     * 
     */
    void render_points_volume(tcb::span<const Vertex> points, float box_size, size_t num_slices, tcb::span<float> result, std::function<bool()> const& should_stop = util::always_false);

    size_t grid_size() const noexcept { return grid_size_; }
};

} // namespace vulkan
} // namespace wenda
