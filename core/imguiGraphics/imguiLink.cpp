#include "imguiLink.h"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

const VkRenderPass &imguiLink::getRenderPass(){
    return renderPass;
}

void imguiLink::setDeviceProp(VkDevice device){
    this->device = device;
}

void imguiLink::setImageCount(const uint32_t& count){
    imageCount = count;
}

void imguiLink::draw(VkCommandBuffer commandBuffer, uint32_t) const {
    // Rendering
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
}

void imguiLink::setRenderPass(VkRenderPass renderPass) {
    this->renderPass = renderPass;
}
