#include "imguiGraphics.h"

#include <string>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "operations.h"
#include "swapChain.h"

imguiGraphics::imguiGraphics(){
    link = &Link;
}

imguiGraphics::~imguiGraphics(){
    imguiGraphics::destroy();
}

void imguiGraphics::destroy() {
    if(descriptorPool){
        vkDestroyDescriptorPool(device.getLogical(), descriptorPool, VK_NULL_HANDLE);
        descriptorPool = VK_NULL_HANDLE;
    }

    if(commandPool) {
        vkDestroyCommandPool(device.getLogical(), commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void imguiGraphics::setInstance(VkInstance instance){
    this->instance = instance;
}

void imguiGraphics::setupImguiContext(){
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
}

void imguiGraphics::createDescriptorPool(){
    std::vector<VkDescriptorPoolSize> descriptorPoolSize = {{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }};
    VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = static_cast<uint32_t>(descriptorPoolSize.size());
        pool_info.pPoolSizes = descriptorPoolSize.data();
    CHECK(vkCreateDescriptorPool(device.getLogical(), &pool_info, VK_NULL_HANDLE, &descriptorPool));
}

void imguiGraphics::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK(vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool));
}

void imguiGraphics::uploadFonts()
{
    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void imguiGraphics::create() {
    setupImguiContext();
    createDescriptorPool();
    createCommandPool();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(swapChainKHR->getWindow(), true);
    ImGui_ImplVulkan_InitInfo initInfo = {};
        initInfo.Instance = instance;
        initInfo.PhysicalDevice = device.instance;
        initInfo.Device = device.getLogical();
        initInfo.QueueFamily = 0;
        initInfo.Queue = device.getQueue(0,0);
        initInfo.PipelineCache = VK_NULL_HANDLE;
        initInfo.DescriptorPool = descriptorPool;
        initInfo.Subpass = 0;
        initInfo.MinImageCount = imageCount;
        initInfo.ImageCount = imageCount;
        initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.Allocator = VK_NULL_HANDLE;
        initInfo.CheckVkResultFn = [](VkResult result){debug::checkResult(result,"");};
    ImGui_ImplVulkan_Init(&initInfo, Link.getRenderPass());

    uploadFonts();
}

void imguiGraphics::update(uint32_t) {}

std::vector<std::vector<VkSemaphore>> imguiGraphics::submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>&, uint32_t){
    return externalSemaphore;
}
