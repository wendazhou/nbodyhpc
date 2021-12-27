#include "point_renderer.h"

#include "shaders/triangle.frag.spv.h"
#include "shaders/triangle.vert.spv.h"

#include <map>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>

#include "thread_pool.hpp"

namespace wenda {
namespace vulkan {

class PointRendererImpl {
  public:
    wenda::vulkan::MemoryBackedImageView color_target_;
    wenda::vulkan::MemoryBackedImage readout_image_;
    vk::raii::RenderPass render_pass_;
    vk::raii::Framebuffer framebuffer_;
    vk::raii::PipelineLayout pipeline_layout_;
    vk::raii::ShaderModule vertex_shader_;
    vk::raii::ShaderModule fragment_shader_;
    vk::raii::DescriptorSetLayout descriptor_set_layout_;
    vk::raii::Pipeline pipeline_;

    vk::raii::CommandBuffer command_buffer_;
};

struct PointRenderingPushConstants {
    float box_size;
    float grid_size;
    float plane_depth;
};

namespace {

vk::raii::RenderPass create_render_pass(vk::raii::Device const &device, vk::Format colorFormat) {
    // Color attachment
    vk::AttachmentDescription attachmentDescription{
        .format = colorFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eTransferSrcOptimal,
    };

    vk::AttachmentReference colorReference{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    vk::SubpassDescription subpassDescription{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorReference,
    };

    vk::SubpassDependency dependencies[2] = {
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eMemoryRead,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead |
                             vk::AccessFlagBits::eColorAttachmentWrite,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        },
        {
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead |
                             vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        }};

    vk::RenderPassCreateInfo renderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = &attachmentDescription,
        .subpassCount = 1,
        .pSubpasses = &subpassDescription,
        .dependencyCount = 2,
        .pDependencies = dependencies,
    };

    return vk::raii::RenderPass(device, renderPassCreateInfo);
}

PointRendererImpl make_point_renderer(VulkanContainer const &container, uint32_t size) {
    auto &device = container.device_;
    auto memory_properties = container.physical_device_.getMemoryProperties();

    uint32_t width = size;
    uint32_t height = size;

    vk::ImageCreateInfo renderTargetInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32Sfloat,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    wenda::vulkan::MemoryBackedImageView colorTarget(
        device,
        renderTargetInfo,
        {
            .viewType = vk::ImageViewType::e2D,
            .format = renderTargetInfo.format,
            .subresourceRange =
                {.aspectMask = vk::ImageAspectFlagBits::eColor,
                 .baseMipLevel = 0,
                 .levelCount = 1,
                 .baseArrayLayer = 0,
                 .layerCount = 1},
        },
        memory_properties,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::raii::RenderPass renderPass = create_render_pass(device, renderTargetInfo.format);

    // Create framebuffer
    vk::FramebufferCreateInfo framebufferCreateInfo{
        .renderPass = *renderPass,
        .attachmentCount = 1,
        .pAttachments = &(*colorTarget.view_),
        .width = width,
        .height = height,
        .layers = 1,
    };

    vk::raii::Framebuffer framebuffer(device, framebufferCreateInfo);

    // create image for host copy
    renderTargetInfo.usage = vk::ImageUsageFlagBits::eTransferDst;
    renderTargetInfo.tiling = vk::ImageTiling::eLinear;

    wenda::vulkan::MemoryBackedImage dstTarget(
        device,
        renderTargetInfo,
        memory_properties,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // load shaders
    vk::raii::ShaderModule vertexShader(
        device,
        {
            .codeSize = triangle_vert_spv_len,
            .pCode = reinterpret_cast<uint32_t const *>(triangle_vert_spv),
        });

    vk::raii::ShaderModule fragmentShader(
        device,
        {
            .codeSize = triangle_frag_spv_len,
            .pCode = reinterpret_cast<uint32_t const *>(triangle_frag_spv),
        });

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        {
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *vertexShader,
            .pName = "main",
        },
        {
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *fragmentShader,
            .pName = "main",
        }};

    // prepare graphics pipeline
    vk::raii::DescriptorSetLayout descriptorSetLayout(
        device,
        {
            .bindingCount = 0,
            .pBindings = nullptr,
        });

    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = sizeof(PointRenderingPushConstants),
    };

    vk::raii::PipelineLayout pipelineLayout(
        device,
        {.setLayoutCount = 1,
         .pSetLayouts = &*descriptorSetLayout,
         .pushConstantRangeCount = 1,
         .pPushConstantRanges = &pushConstantRange});

    vk::VertexInputBindingDescription vertexInputBindings[] = {{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    }};

    vk::VertexInputAttributeDescription vertexInputAttributes[] = {
        {
            // position
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, position),
        },
        {
            // weight
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32Sfloat,
            .offset = offsetof(Vertex, weight),
        },
        {
            // radius
            .location = 2,
            .binding = 0,
            .format = vk::Format::eR32Sfloat,
            .offset = offsetof(Vertex, radius),
        }};

    vk::PipelineVertexInputStateCreateInfo vertexInputState{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = vertexInputBindings,
        .vertexAttributeDescriptionCount =
            sizeof(vertexInputAttributes) / sizeof(vertexInputAttributes[0]),
        .pVertexAttributeDescriptions = vertexInputAttributes,
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{
        .topology = vk::PrimitiveTopology::ePointList,
        .primitiveRestartEnable = VK_FALSE,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizationState{
        .depthClampEnable = VK_FALSE,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };

    vk::PipelineColorBlendAttachmentState blendAttachmentState{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eOne,
        .colorBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &blendAttachmentState,
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
    };

    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    std::array<vk::DynamicState, 2> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = dynamicStates.size(),
        .pDynamicStates = dynamicStates.data(),
    };

    vk::PipelineMultisampleStateCreateInfo multisampleState{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo{
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputState,
        .pInputAssemblyState = &inputAssemblyState,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizationState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &depthStencilState,
        .pColorBlendState = &colorBlendState,
        .pDynamicState = &dynamicState,
        .layout = *pipelineLayout,
        .renderPass = *renderPass,
    };

    vk::raii::Pipeline pipeline(device, nullptr, pipelineCreateInfo, nullptr);

    vk::raii::CommandBuffers commandBuffers(
        device, {.commandPool = *container.command_pool_, .commandBufferCount = 1});

    return PointRendererImpl{
        .color_target_ = std::move(colorTarget),
        .readout_image_ = std::move(dstTarget),
        .render_pass_ = std::move(renderPass),
        .framebuffer_ = std::move(framebuffer),
        .pipeline_layout_ = std::move(pipelineLayout),
        .vertex_shader_ = std::move(vertexShader),
        .fragment_shader_ = std::move(fragmentShader),
        .descriptor_set_layout_ = std::move(descriptorSetLayout),
        .pipeline_ = std::move(pipeline),
        .command_buffer_ = std::move(commandBuffers[0]),
    };
}

/** Submit work to the given queue, and block until it is complete.
 * 
 */
void submit_work(vk::raii::Device const& device, vk::raii::Queue const& queue, vk::SubmitInfo info) {
    auto fence = device.createFence({});

    queue.submit(
        {info},
        *fence);

    auto wait_result = device.waitForFences(
        {
            *fence,
        },
        VK_TRUE,
        std::numeric_limits<uint64_t>::max());
}

/** Copies the vertex data into a vertex buffer, staging the data through a temporary staging buffer.
 * 
 */
std::tuple<vk::raii::Buffer, vk::raii::DeviceMemory> stage_to_vertex_buffer(VulkanContainer const& container, tcb::span<const Vertex> vertices) {
    vk::DeviceSize vertexBufferSize = vertices.size_bytes();

    vk::raii::Buffer vertexBuffer(
        container.device_,
        {.size = vertexBufferSize, .usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst});

    vk::raii::Buffer staging_buffer(
        container.device_,
        {.size = vertexBufferSize, .usage = vk::BufferUsageFlagBits::eTransferSrc});

    auto memory_properties = container.physical_device_.getMemoryProperties();
    auto staging_memory_requirements = staging_buffer.getMemoryRequirements();
    auto vertexMemoryRequirements = vertexBuffer.getMemoryRequirements();

    vk::raii::DeviceMemory staging_memory(
        container.device_,
        {
            .allocationSize = staging_memory_requirements.size,
            .memoryTypeIndex = wenda::vulkan::determine_memory_type_index(
                staging_memory_requirements,
                memory_properties,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
        });
    staging_buffer.bindMemory(*staging_memory, 0);

    vk::raii::DeviceMemory vertex_buffer_memory(
        container.device_,
        {.allocationSize = vertexMemoryRequirements.size,
         .memoryTypeIndex = wenda::vulkan::determine_memory_type_index(
             vertexMemoryRequirements,
             memory_properties,
             vk::MemoryPropertyFlagBits::eDeviceLocal)});
    vertexBuffer.bindMemory(*vertex_buffer_memory, 0);

    wenda::vulkan::copy_to_device_memory(vertices.begin(), vertices.end(), staging_memory);

    vk::raii::CommandBuffers copy_command_buffers(
        container.device_,
        {.commandPool = *container.command_pool_, .commandBufferCount = 1});
    auto& command_buffer = copy_command_buffers[0];

    command_buffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    command_buffer.copyBuffer(
        *staging_buffer,
        *vertexBuffer,
        {vk::BufferCopy{.size = vertexBufferSize}});
    command_buffer.end();

    submit_work(container.device_, container.queue_, {.commandBufferCount = 1, .pCommandBuffers = &(*command_buffer)});

    return std::make_tuple(std::move(vertexBuffer), std::move(vertex_buffer_memory));
}

template<typename It>
void read_buffer_strided(vk::raii::DeviceMemory const& memory, It output, uint32_t grid_size, vk::SubresourceLayout const& layout) {
    typedef typename std::iterator_traits<It>::value_type T;

    auto row_pitch_elements = layout.rowPitch / sizeof(T);

    T* data = static_cast<T *>(memory.mapMemory(0, VK_WHOLE_SIZE)) + layout.offset;

    for (int y = 0; y < grid_size; ++y) {
        std::copy(
            data + y * row_pitch_elements,
            data + y * row_pitch_elements + grid_size,
            output + y * grid_size);
    }

    memory.unmapMemory();
}

/** Builds up the command buffer to render the point cloud.
 * 
 */
void build_point_render_commands(vk::raii::CommandBuffer& command_buffer, PointRendererImpl const& renderer, vk::Buffer const& vertex_buffer, uint32_t num_vertices, uint32_t grid_size, float box_size, float plane_depth) {
    uint32_t width = grid_size;
    uint32_t height = grid_size;

    std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 1.0f};

    vk::ClearValue clearValues[] = {
        {
            .color = {clearColor},
        },
    };

    command_buffer.beginRenderPass(
        {
            .renderPass = *renderer.render_pass_,
            .framebuffer = *renderer.framebuffer_,
            .renderArea =
                {
                    .offset = {0, 0},
                    .extent = {width, height},
                },
            .clearValueCount = 1,
            .pClearValues = clearValues,
        },
        vk::SubpassContents::eInline);

    command_buffer.setViewport(
        0,
        {{
            .x = 0,
            .y = 0,
            .width = static_cast<float>(width),
            .height = static_cast<float>(height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        }});

    command_buffer.setScissor(
        0,
        {{
            .offset = {0, 0},
            .extent = {width, height},
        }});

    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *renderer.pipeline_);

    command_buffer.bindVertexBuffers(0, {vertex_buffer}, {0});

    PointRenderingPushConstants push_constants_data{
        .box_size = box_size,
        .grid_size = static_cast<float>(width),
        .plane_depth = plane_depth,
    };

    command_buffer.pushConstants<char>(
        *renderer.pipeline_layout_,
        vk::ShaderStageFlagBits::eVertex,
        0,
        {sizeof(push_constants_data), reinterpret_cast<char *>(&push_constants_data)});
    command_buffer.draw(num_vertices, 1, 0, 0);

    command_buffer.endRenderPass();
}

void build_image_transfer_commands(vk::raii::CommandBuffer& command_buffer, PointRendererImpl const& renderer, MemoryBackedImage const& readout_image, uint32_t grid_size) {
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlags(),
        {},
        {},
        {vk::ImageMemoryBarrier{
            .srcAccessMask = vk::AccessFlagBits(),
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = *readout_image.image_,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        }});

    command_buffer.copyImage(
        *renderer.color_target_.image_,
        vk::ImageLayout::eTransferSrcOptimal,
        *readout_image.image_,
        vk::ImageLayout::eTransferDstOptimal,
        {vk::ImageCopy{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .extent = {grid_size, grid_size, 1},
        }});

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlags(),
        {},
        {},
        {vk::ImageMemoryBarrier{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = *readout_image.image_,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }});
}


} // namespace

PointRenderer::PointRenderer(VulkanContainer const &container, float box_size, uint32_t grid_size)
    : impl_(std::make_unique<PointRendererImpl>(make_point_renderer(container, grid_size))),
      container_(container), box_size_(box_size), grid_size_(grid_size) {}

PointRenderer::~PointRenderer() {}

void PointRenderer::render_points(tcb::span<const Vertex> points, tcb::span<float> result) {
    if (result.size() < grid_size_ * grid_size_) {
        throw std::runtime_error("result buffer too small");
    }

    // set-up vertex buffer
    auto const [vertex_buffer, vertex_buffer_memory] = stage_to_vertex_buffer(container_, points);

    auto &command_buffer = impl_->command_buffer_;
    uint32_t width = grid_size_;
    uint32_t height = grid_size_;

    command_buffer.begin({});

    build_point_render_commands(command_buffer, *impl_, *vertex_buffer, points.size(), grid_size_, box_size_, 0.0f);
    build_image_transfer_commands(command_buffer, *impl_, impl_->readout_image_, grid_size_);

    command_buffer.end();

    submit_work(container_.device_, container_.queue_, {.commandBufferCount = 1, .pCommandBuffers = &(*command_buffer)});

    vk::SubresourceLayout dstImageLayout =
        impl_->readout_image_.image_.getSubresourceLayout({vk::ImageAspectFlagBits::eColor, 0, 0});
    read_buffer_strided(impl_->readout_image_.memory_, result.data(), grid_size_, dstImageLayout);
}

namespace {

/** Thread-safe collection of transfer images.
 * This can be used to copy from different target images in a round-robin fashion.
 * 
 */
class TransferImagePool {
    std::queue<MemoryBackedImage> images_;
    std::mutex mutex_;
    std::condition_variable cv_;

public:
    TransferImagePool(vk::raii::Device const &device, vk::PhysicalDeviceMemoryProperties const& memory_properties, uint32_t grid_size, uint32_t num_images)
    {
        vk::ImageCreateInfo transfer_image_info{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR32Sfloat,
            .extent = {grid_size, grid_size, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eLinear,
            .usage = vk::ImageUsageFlagBits::eTransferDst,
            .initialLayout = vk::ImageLayout::eUndefined,
        };

        for (uint32_t i = 0; i < num_images; ++i) {
            images_.emplace(device, transfer_image_info, memory_properties, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        }
    }

    MemoryBackedImage get_image() {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);

        while (true) {
            if (!images_.empty()) {
                auto image = std::move(images_.front());
                images_.pop();
                return image;
            }

            cv_.wait(lock);
        }
    }

    void return_image(MemoryBackedImage&& image) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            images_.push(std::move(image));
        }
        cv_.notify_one();
    }
};

}

void PointRenderer::render_points_volume(tcb::span<const Vertex> points, tcb::span<float> result, std::function<bool()> const& should_stop) {
    if (result.size() < grid_size_ * grid_size_ * grid_size_) {
        throw std::runtime_error("result buffer too small");
    }

    TransferImagePool transfer_images(container_.device_, container_.physical_device_.getMemoryProperties(), grid_size_, 4);
    thread_pool pool(4);

    std::map<int, MemoryBackedImage> transfer_images_map;
    std::mutex transfer_images_map_mutex;

    // set-up vertex buffer
    auto const [vertexBuffer, vertexBufferMemory] = stage_to_vertex_buffer(container_, points);

    auto &command_buffer = impl_->command_buffer_;


    for (int i = 0; i < grid_size_; ++i) {
        if (should_stop()) {
            break;
        }

        auto transfer_image = transfer_images.get_image();

        command_buffer.begin({});

        build_point_render_commands(command_buffer, *impl_, *vertexBuffer, points.size(), grid_size_, box_size_,
                                    ((static_cast<float>(i) + 0.5f) / grid_size_) * box_size_);
        build_image_transfer_commands(command_buffer, *impl_, transfer_image, grid_size_);

        command_buffer.end();

        submit_work(container_.device_, container_.queue_, {.commandBufferCount = 1, .pCommandBuffers = &(*command_buffer)});

        {
            std::scoped_lock lock(transfer_images_map_mutex);
            transfer_images_map.insert({i, std::move(transfer_image)});
        }

        pool.push_task([&, i] () {
            std::optional<MemoryBackedImage> transfer_image;

            {
                std::scoped_lock lock(transfer_images_map_mutex);
                auto transfer_image_it = transfer_images_map.find(i);
                transfer_image = std::move(transfer_image_it->second);
                transfer_images_map.erase(transfer_image_it);
            }

            vk::SubresourceLayout dstImageLayout =
                transfer_image->image_.getSubresourceLayout({vk::ImageAspectFlagBits::eColor, 0, 0});
            read_buffer_strided(transfer_image->memory_, result.data() + i * grid_size_ * grid_size_, grid_size_,
                                dstImageLayout);
            transfer_images.return_image(std::move(*transfer_image));
        });
    }

    pool.wait_for_tasks();
}

namespace util {
    bool always_false() noexcept {
        return false;
    }
}

} // namespace vulkan
} // namespace wenda
