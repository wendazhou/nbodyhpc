#include "vulkan_support.h"

#include <iostream>
#include <sstream>

namespace wenda {
namespace vulkan {

namespace {
std::tuple<vk::raii::Image, vk::raii::DeviceMemory> create_memory_backed_image(
    vk::raii::Device const &device, vk::ImageCreateInfo const &image_info,
    vk::PhysicalDeviceMemoryProperties const &memory_properties,
    vk::MemoryPropertyFlags memory_property_flags) {
    vk::raii::Image image(device, image_info);
    auto memory_requirements = image.getMemoryRequirements();
    vk::raii::DeviceMemory memory(
        device,
        {.allocationSize = memory_requirements.size,
         .memoryTypeIndex = wenda::vulkan::determine_memory_type_index(
             memory_requirements, memory_properties, memory_property_flags)});

    image.bindMemory(*memory, 0);
    return std::make_tuple(std::move(image), std::move(memory));
}
} // namespace

MemoryBackedImage::MemoryBackedImage(
    vk::raii::Device const &device, vk::ImageCreateInfo const &image_info,
    vk::PhysicalDeviceMemoryProperties const &memory_properties,
    vk::MemoryPropertyFlags memory_property_flags)
    : MemoryBackedImage(create_memory_backed_image(
          device, image_info, memory_properties, memory_property_flags)) {}

MemoryBackedImage::MemoryBackedImage(std::tuple<vk::raii::Image, vk::raii::DeviceMemory> data)
    : image_(std::move(std::get<0>(data))), memory_(std::move(std::get<1>(data))) {}

MemoryBackedImageView::MemoryBackedImageView(
    vk::raii::Device const &device, vk::ImageCreateInfo const &image_info,
    vk::ImageViewCreateInfo const &view_info,
    vk::PhysicalDeviceMemoryProperties const &memory_properties,
    vk::MemoryPropertyFlags memory_property_flags)
    : MemoryBackedImage(device, image_info, memory_properties, memory_property_flags),
      view_(vk::raii::ImageView(device, [this](vk::ImageViewCreateInfo info) {
          info.image = *image_;
          return info;
      }(view_info))) {}

uint32_t determine_memory_type_index(
    vk::MemoryRequirements const &memoryRequirements,
    vk::PhysicalDeviceMemoryProperties const &memoryProperties,
    vk::MemoryPropertyFlags memoryPropertyFlags) {
    uint32_t typeIndex = uint32_t(~0);
    uint32_t typeBits = memoryRequirements.memoryTypeBits;

    bool found = false;

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) && ((memoryProperties.memoryTypes[i].propertyFlags &
                                memoryPropertyFlags) == memoryPropertyFlags)) {
            typeIndex = i;
            found = true;
            break;
        }
        typeBits >>= 1;
    }

    if (!found) {
        throw std::runtime_error("Could not find suitable memory type");
    }

    return typeIndex;
}

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_message_func(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    VkDebugUtilsMessengerCallbackDataEXT const *pCallbackData, void * /*pUserData*/) {
    std::ostringstream message;

    message << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity))
            << ": " << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes))
            << ":\n";
    message << "\t"
            << "messageIDName   = <" << pCallbackData->pMessageIdName << ">\n";
    message << "\t"
            << "messageIdNumber = " << pCallbackData->messageIdNumber << "\n";
    message << "\t"
            << "message         = <" << pCallbackData->pMessage << ">\n";
    if (0 < pCallbackData->queueLabelCount) {
        message << "\t"
                << "Queue Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->queueLabelCount; i++) {
            message << "\t\t"
                    << "labelName = <" << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->cmdBufLabelCount) {
        message << "\t"
                << "CommandBuffer Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->cmdBufLabelCount; i++) {
            message << "\t\t"
                    << "labelName = <" << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->objectCount) {
        message << "\t"
                << "Objects:\n";
        for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
            message << "\t\t"
                    << "Object " << i << "\n";
            message << "\t\t\t"
                    << "objectType   = "
                    << vk::to_string(
                           static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType))
                    << "\n";
            message << "\t\t\t"
                    << "objectHandle = " << pCallbackData->pObjects[i].objectHandle << "\n";
            if (pCallbackData->pObjects[i].pObjectName) {
                message << "\t\t\t"
                        << "objectName   = <" << pCallbackData->pObjects[i].pObjectName << ">\n";
            }
        }
    }

    std::cout << message.str() << std::endl;

    return false;
}

vk::raii::DebugUtilsMessengerEXT
create_debug_util_messenger_extension(vk::raii::Instance const &instance) {
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
        .messageSeverity = severityFlags,
        .messageType = messageTypeFlags,
        .pfnUserCallback = vulkan_debug_message_func,
        .pUserData = nullptr};
    return vk::raii::DebugUtilsMessengerEXT(instance, debugUtilsMessengerCreateInfoEXT);
}

VulkanContainerFields initialize_vulkan(bool enable_validation_layers) {
    // Create Vulkan instance, context + device
    vk::raii::Context context;

    vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0};

    const char *validationLayers[] = {"VK_LAYER_KHRONOS_validation"};
    uint32_t layerCount = 0;

    if (enable_validation_layers) {
        layerCount = 1;
    }

    const char *extensions[] = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    uint32_t extensionCount = 0;

    if (enable_validation_layers) {
        extensionCount = 1;
    }

    vk::InstanceCreateInfo instanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = layerCount,
        .ppEnabledLayerNames = validationLayers,
        .enabledExtensionCount = extensionCount,
        .ppEnabledExtensionNames = extensions};

    vk::raii::Instance instance{context, instanceCreateInfo};

    std::optional<vk::raii::DebugUtilsMessengerEXT> debugUtilsMessenger;

    if (enable_validation_layers) {
        debugUtilsMessenger = create_debug_util_messenger_extension(instance);
    }

    vk::raii::PhysicalDevices physicalDevices{instance};

    auto physicalDevice = std::move(physicalDevices[0]);

    auto const &queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    auto it_graphics_queue = std::find_if(
        queueFamilyProperties.begin(),
        queueFamilyProperties.end(),
        [](vk::QueueFamilyProperties const &qfp) {
            return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
        });

    auto it_transfer_queue = std::find_if(
        queueFamilyProperties.begin(),
        queueFamilyProperties.end(),
        [](vk::QueueFamilyProperties const &qfp) {
            return (qfp.queueFlags & vk::QueueFlagBits::eTransfer) &&
                   !(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

    uint32_t graphicsQueueFamilyIndex =
        static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), it_graphics_queue));

    float queuePriority = 1.0f;

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    uint32_t num_graphics_queues = it_graphics_queue->queueCount;
    std::vector<float> queuePriorities(num_graphics_queues, queuePriority);

    queueCreateInfos.push_back(
        {.queueFamilyIndex = graphicsQueueFamilyIndex,
         .queueCount = num_graphics_queues,
         .pQueuePriorities = queuePriorities.data()});


    uint32_t transferQueueFamilyIndex = graphicsQueueFamilyIndex;
    if (it_transfer_queue != queueFamilyProperties.end()) {
        // if there is a separate transfer queue, make use of it
        transferQueueFamilyIndex =
            static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), it_transfer_queue));
        queueCreateInfos.push_back(
            {.queueFamilyIndex = transferQueueFamilyIndex,
             .queueCount = 1,
             .pQueuePriorities = queuePriorities.data()});
    }

    vk::PhysicalDeviceFeatures deviceFeatures{
        .shaderClipDistance = true,
    };

    vk::DeviceCreateInfo deviceCreateInfo{
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = layerCount,
        .ppEnabledLayerNames = validationLayers,
        .pEnabledFeatures = &deviceFeatures,
    };

    vk::raii::Device device(physicalDevice, deviceCreateInfo);

    std::vector<vk::raii::Queue> graphics_queues;
    for(uint32_t i = 0; i < num_graphics_queues; ++i) {
        graphics_queues.push_back(vk::raii::Queue(device, graphicsQueueFamilyIndex, i));
    }
    vk::raii::Queue transfer_queue{device, transferQueueFamilyIndex, 0};

    // Create command pool and buffers
    std::vector<vk::raii::CommandPool> commandPools;
    for(uint32_t i = 0; i < num_graphics_queues; ++i) {
        commandPools.push_back(vk::raii::CommandPool(device, {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = graphicsQueueFamilyIndex,
        }));
    }

    vk::raii::CommandPool transferCommandPool(
        device,
        {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = transferQueueFamilyIndex,
        });

    return {
        std::move(context),
        std::move(instance),
        std::move(physicalDevice),
        std::move(device),
        std::move(graphics_queues),
        std::move(transfer_queue),
        std::move(commandPools),
        std::move(transferCommandPool),
        std::move(debugUtilsMessenger)
    };
}

} // namespace

VulkanContainer::VulkanContainer(bool enable_validation_layers)
    : VulkanContainer(initialize_vulkan(enable_validation_layers)) {}
VulkanContainer::VulkanContainer(VulkanContainerFields&& fields) noexcept : VulkanContainerFields(std::move(fields)) {}

} // namespace vulkan
} // namespace wenda