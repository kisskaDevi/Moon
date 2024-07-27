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
    setupImguiContext();
    link = std::make_unique<ImguiLink>();
}

ImguiGraphics::~ImguiGraphics(){
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

void ImguiGraphics::reset() {
    std::vector<VkDescriptorPoolSize> descriptorPoolSize = { VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSize.size());
        poolInfo.pPoolSizes = descriptorPoolSize.data();
    descriptorPool = utils::vkDefault::DescriptorPool(device->getLogical(), poolInfo);
    commandPool = utils::vkDefault::CommandPool(device->getLogical());

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
    ImGui_ImplVulkan_Init(&initInfo, link->renderPass());

    uploadFonts();
}

void ImguiGraphics::update(uint32_t) {}

utils::vkDefault::VkSemaphores ImguiGraphics::submit(uint32_t, const utils::vkDefault::VkSemaphores& externalSemaphore) {
    return externalSemaphore;
}

}
