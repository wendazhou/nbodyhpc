#pragma once

#include <algorithm>
#include <optional>
#include <tuple>
#include <type_traits>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace wenda {

namespace vulkan {

struct VulkanContainerFields {
    vk::raii::Context context_;
    vk::raii::Instance instance_;
    vk::raii::PhysicalDevice physical_device_;
    vk::raii::Device device_;
    vk::raii::Queue queue_;
    vk::raii::Queue transfer_queue_;
    vk::raii::CommandPool command_pool_;
    vk::raii::CommandPool transfer_command_pool_;
    std::optional<vk::raii::DebugUtilsMessengerEXT> debug_messenger_;
};

/** Main class containing the vulkan objects that are persistent throughout the application.
 *
 */
class VulkanContainer : public VulkanContainerFields {
  protected:
    VulkanContainer(VulkanContainerFields&& fields) noexcept;

  public:
    explicit VulkanContainer(bool enable_validation_layers = true);
    VulkanContainer(VulkanContainer&&) noexcept = default;
};

/** Useful wrapper for creating an image, allocating memory for it, and binding it to a view.
 *
 */
struct MemoryBackedImage {
    vk::raii::Image image_;
    vk::raii::DeviceMemory memory_;

    MemoryBackedImage(
        vk::raii::Device const &device, vk::ImageCreateInfo const &image_info,
        vk::PhysicalDeviceMemoryProperties const &memory_properties,
        vk::MemoryPropertyFlags memory_property_flags);

  private:
    MemoryBackedImage(std::tuple<vk::raii::Image, vk::raii::DeviceMemory>);
};

struct MemoryBackedImageView : MemoryBackedImage {
    vk::raii::ImageView view_;

    MemoryBackedImageView(
        vk::raii::Device const &device, vk::ImageCreateInfo const &image_info,
        vk::ImageViewCreateInfo const &view_info,
        vk::PhysicalDeviceMemoryProperties const &memory_properties,
        vk::MemoryPropertyFlags memory_property_flags);
};

/** Utility function for finding a memory heap which satisfies the given requirements.
 *
 * @param memory_requirements The memory requirements of the object to be allocated.
 * @param memory_properties The memory properties of the different heaps in physical device.
 * @param flags Additional memory property flags that the heap must satisfy.
 *
 * @return The index of the heap that satisfies the requirements.
 *
 */
uint32_t determine_memory_type_index(
    vk::MemoryRequirements const &memory_requirements,
    vk::PhysicalDeviceMemoryProperties const &memory_properties,
    vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

template <typename It> void copy_to_device_memory(It begin, It end, vk::raii::DeviceMemory &memory) {
    auto dst = static_cast<std::remove_const_t<typename std::iterator_traits<It>::value_type>*>(
        memory.mapMemory(0, sizeof(*begin) * std::distance(begin, end)));
    std::copy(begin, end, dst);
    memory.unmapMemory();
}

} // namespace vulkan
} // namespace wenda
