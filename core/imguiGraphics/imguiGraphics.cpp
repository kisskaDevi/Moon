#include "imguiGraphics.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "operations.h"

namespace moon::imguiGraphics {

ImguiGraphics::ImguiGraphics(GLFWwindow* window, VkInstance instance, uint32_t imageCount) :
    window(window),
    instance(instance),
    imageCount(imageCount) {
    link = &Link;
}

ImguiGraphics::~ImguiGraphics(){
    ImguiGraphics::destroy();
}

void ImguiGraphics::destroy() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImguiGraphics::setupImguiContext(){
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
}

void ImguiGraphics::uploadFonts()
{
    VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device->getLogical(),commandPool);
    ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
    moon::utils::singleCommandBuffer::submit(device->getLogical(),device->getQueue(0,0),commandPool,&commandBuffer);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void ImguiGraphics::create() {
    std::vector<VkDescriptorPoolSize> descriptorPoolSize = { VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSize.size());
        poolInfo.pPoolSizes = descriptorPoolSize.data();
    CHECK(descriptorPool.create(device->getLogical(), poolInfo));
    CHECK(commandPool.create(device->getLogical()));

    setupImguiContext();
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo initInfo = {};
        initInfo.Instance = instance;
        initInfo.PhysicalDevice = device->instance;
        initInfo.Device = device->getLogical();
        initInfo.QueueFamily = 0;
        initInfo.Queue = device->getQueue(0,0);
        initInfo.PipelineCache = VK_NULL_HANDLE;
        initInfo.DescriptorPool = descriptorPool;
        initInfo.Subpass = 0;
        initInfo.MinImageCount = imageCount;
        initInfo.ImageCount = imageCount;
        initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.Allocator = VK_NULL_HANDLE;
        initInfo.CheckVkResultFn = [](VkResult result){moon::utils::debug::checkResult(result,"");};
    ImGui_ImplVulkan_Init(&initInfo, Link.getRenderPass());

    uploadFonts();
}

void ImguiGraphics::update(uint32_t) {}

std::vector<std::vector<VkSemaphore>> ImguiGraphics::submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>&, uint32_t){
    return externalSemaphore;
}

}
