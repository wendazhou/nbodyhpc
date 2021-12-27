#include "vulkan_support.h"

#include <iostream>
#include <sstream>

namespace wenda { namespace vulkan {

namespace {
    std::tuple<vk::raii::Image, vk::raii::DeviceMemory> create_memory_backed_image(
        vk::raii::Device const& device, vk::ImageCreateInfo const& image_info,
        vk::PhysicalDeviceMemoryProperties const& memory_properties,
        vk::MemoryPropertyFlags memory_property_flags)
    {
        vk::raii::Image image(device, image_info);
        auto memory_requirements = image.getMemoryRequirements();
        vk::raii::DeviceMemory memory(device, {
            .allocationSize = memory_requirements.size,
            .memoryTypeIndex = wenda::vulkan::determine_memory_type_index(
                memory_requirements, memory_properties, memory_property_flags)
        });

        image.bindMemory(*memory, 0);
        return std::make_tuple(std::move(image), std::move(memory));
    }
}

MemoryBackedImage::MemoryBackedImage(
        vk::raii::Device const &device, vk::ImageCreateInfo const &image_info,
        vk::PhysicalDeviceMemoryProperties const &memory_properties,
        vk::MemoryPropertyFlags memory_property_flags)
        : MemoryBackedImage(create_memory_backed_image(device, image_info, memory_properties, memory_property_flags)) {}

MemoryBackedImage::MemoryBackedImage(std::tuple<vk::raii::Image, vk::raii::DeviceMemory> data)
: image_(std::move(std::get<0>(data))), memory_(std::move(std::get<1>(data))) {}


MemoryBackedImageView::MemoryBackedImageView(
        vk::raii::Device const &device, vk::ImageCreateInfo const& image_info,
        vk::ImageViewCreateInfo const &view_info,
        vk::PhysicalDeviceMemoryProperties const &memory_properties,
        vk::MemoryPropertyFlags memory_property_flags)
    : MemoryBackedImage(device, image_info, memory_properties, memory_property_flags),
    view_(vk::raii::ImageView(device, [this](vk::ImageViewCreateInfo info) {
        info.image = *image_;
        return info;
    }(view_info))) {}


uint32_t determine_memory_type_index(vk::MemoryRequirements const &memoryRequirements,
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


VKAPI_ATTR VkBool32 VKAPI_CALL
vulkan_debug_message_func(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                 VkDebugUtilsMessengerCallbackDataEXT const *pCallbackData,
                 void * /*pUserData*/) {
    std::ostringstream message;

    message << vk::to_string(
                   static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(
                       messageSeverity))
            << ": "
            << vk::to_string(
                   static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes))
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
                    << "labelName = <"
                    << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->cmdBufLabelCount) {
        message << "\t"
                << "CommandBuffer Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->cmdBufLabelCount; i++) {
            message << "\t\t"
                    << "labelName = <"
                    << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
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
                    << vk::to_string(static_cast<vk::ObjectType>(
                           pCallbackData->pObjects[i].objectType))
                    << "\n";
            message << "\t\t\t"
                    << "objectHandle = "
                    << pCallbackData->pObjects[i].objectHandle << "\n";
            if (pCallbackData->pObjects[i].pObjectName) {
                message << "\t\t\t"
                        << "objectName   = <"
                        << pCallbackData->pObjects[i].pObjectName << ">\n";
            }
        }
    }

    std::cout << message.str() << std::endl;

    return false;
}


vk::raii::DebugUtilsMessengerEXT create_debug_util_messenger_extension(vk::raii::Instance const &instance) {
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


std::tuple<vk::raii::Context, vk::raii::Instance, vk::raii::PhysicalDevice, vk::raii::Device, vk::raii::Queue, vk::raii::CommandPool, std::optional<vk::raii::DebugUtilsMessengerEXT>> initialize_vulkan() {
    // Create Vulkan instance, context + device
    vk::raii::Context context;

    vk::ApplicationInfo appInfo{.pApplicationName = "Hello Triangle",
                                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                .pEngineName = "No Engine",
                                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                .apiVersion = VK_API_VERSION_1_0};

    const char *validationLayers[] = {"VK_LAYER_KHRONOS_validation"};
    uint32_t layerCount = 1;

    const char *extensions[] = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };

    vk::InstanceCreateInfo instanceCreateInfo{.pApplicationInfo = &appInfo,
                                              .enabledLayerCount = layerCount,
                                              .ppEnabledLayerNames = validationLayers,
                                              .enabledExtensionCount =
                                                  sizeof(extensions) / sizeof(extensions[0]),
                                              .ppEnabledExtensionNames = extensions};

    vk::raii::Instance instance{context, instanceCreateInfo};

    vk::raii::DebugUtilsMessengerEXT debugUtilsMessenger = create_debug_util_messenger_extension(instance);

    vk::raii::PhysicalDevices physicalDevices{instance};

    auto physicalDevice = std::move(physicalDevices[0]);

    auto const &queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    auto it_queue = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                                 [](vk::QueueFamilyProperties const &qfp) {
                                     return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
                                 });
    uint32_t graphicsQueueFamilyIndex =
        static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), it_queue));

    float queuePriority = 1.0f;

    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
        .queueFamilyIndex = graphicsQueueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    vk::PhysicalDeviceFeatures deviceFeatures {
        .shaderClipDistance = true,
    };

    vk::DeviceCreateInfo deviceCreateInfo{
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledLayerCount = layerCount,
        .ppEnabledLayerNames = validationLayers,
        .pEnabledFeatures = &deviceFeatures,
    };

    vk::raii::Device device(physicalDevice, deviceCreateInfo);

    vk::raii::Queue queue{device, graphicsQueueFamilyIndex, 0};

    // Create command pool and buffers
    vk::CommandPoolCreateInfo commandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = graphicsQueueFamilyIndex,
    };

    vk::raii::CommandPool commandPool(device, commandPoolCreateInfo);

    return std::make_tuple(std::move(context), std::move(instance), std::move(physicalDevice), std::move(device), std::move(queue), std::move(commandPool), std::move(debugUtilsMessenger));
}

}

VulkanContainer::VulkanContainer() : VulkanContainer(initialize_vulkan()) {}
VulkanContainer::VulkanContainer(std::tuple<vk::raii::Context, vk::raii::Instance, vk::raii::PhysicalDevice,
                    vk::raii::Device, vk::raii::Queue, vk::raii::CommandPool, std::optional<vk::raii::DebugUtilsMessengerEXT>> data) :
    context_(std::move(std::get<0>(data))),
    instance_(std::move(std::get<1>(data))),
    physical_device_(std::move(std::get<2>(data))),
    device_(std::move(std::get<3>(data))),
    queue_(std::move(std::get<4>(data))),
    command_pool_(std::move(std::get<5>(data))),
    debug_messenger_(std::move(std::get<6>(data))) {
    }

}}