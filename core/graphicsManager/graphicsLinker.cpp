#include "graphicsLinker.h"
#include "attachments.h"
#include "swapChain.h"
#include "linkable.h"

namespace moon::graphicsManager {

GraphicsLinker::~GraphicsLinker(){
    destroy();
}

void GraphicsLinker::destroy(){
    if(!commandBuffers.empty()){
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    }
    commandBuffers.clear();

    if(commandPool) {
        vkDestroyCommandPool(device, commandPool, nullptr); commandPool = VK_NULL_HANDLE;
    }
}

void GraphicsLinker::setSwapChain(moon::utils::SwapChain* swapChainKHR){
    this->swapChainKHR = swapChainKHR;
    this->imageCount = swapChainKHR->getImageCount();
    this->imageExtent = swapChainKHR->getExtent();
    this->imageFormat = swapChainKHR->getFormat();
}

void GraphicsLinker::setDevice(VkDevice device){
    this->device = device;
}

void GraphicsLinker::addLinkable(Linkable* link){
    linkables.push_back(link);
    updateCmdFlags();
}

void GraphicsLinker::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(imageFormat, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
        attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    utils::vkDefault::RenderPass::SubpassDescriptions subpasses;
    for(auto refIt = attachmentRef.begin(); refIt != attachmentRef.end(); refIt++){
        subpasses.push_back(VkSubpassDescription{});
        subpasses.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpasses.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
        subpasses.back().pColorAttachments = refIt->data();
    }

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
    dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies.back().dstSubpass = 0;
    dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependencies.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    CHECK(renderPass.create(device, attachments, subpasses, dependencies));
}

void GraphicsLinker::createFramebuffers(){
    framebuffers.resize(imageCount);
    for (size_t i = 0; i < static_cast<uint32_t>(framebuffers.size()); i++) {
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &swapChainKHR->attachment(i).imageView;
            framebufferInfo.width = imageExtent.width;
            framebufferInfo.height = imageExtent.height;
            framebufferInfo.layers = 1;
        CHECK(framebuffers[i].create(device, framebufferInfo));
    }
}

void GraphicsLinker::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
}

void GraphicsLinker::createCommandBuffers(){
    commandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(imageCount);
    CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));

    updateCommandBufferFlags.resize(imageCount, true);
}

void GraphicsLinker::updateCommandBuffer(uint32_t resourceNumber, uint32_t imageNumber){
    if(updateCommandBufferFlags[resourceNumber])
    {
        CHECK(vkResetCommandBuffer(commandBuffers[resourceNumber],0));

        VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;
        CHECK(vkBeginCommandBuffer(commandBuffers[resourceNumber], &beginInfo));

            std::vector<VkClearValue> clearValues = {VkClearValue{}};

            VkRenderPassBeginInfo renderPassInfo{};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = framebuffers[imageNumber];
                renderPassInfo.renderArea.offset = {0,0};
                renderPassInfo.renderArea.extent = imageExtent;
                renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
                renderPassInfo.pClearValues = clearValues.data();

            vkCmdBeginRenderPass(commandBuffers[resourceNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

                for(const auto& link: linkables){
                    link->draw(commandBuffers[resourceNumber], resourceNumber);
                }

            vkCmdEndRenderPass(commandBuffers[resourceNumber]);

        CHECK(vkEndCommandBuffer(commandBuffers[resourceNumber]));

        updateCommandBufferFlags[resourceNumber] = false;
    }
}

void GraphicsLinker::createSyncObjects(){
    signalSemaphores.resize(imageCount);
    for (auto& semaphore: signalSemaphores){
        CHECK(semaphore.create(device));
    }
}

const VkSemaphore& GraphicsLinker::submit(uint32_t frameNumber, const std::vector<VkSemaphore>& waitSemaphores, VkFence fence, VkQueue queue){
    VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = submitInfo.waitSemaphoreCount > 0 ? waitSemaphores.data() : VK_NULL_HANDLE;
        submitInfo.pWaitDstStageMask = &waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[frameNumber];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores[frameNumber];
    CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    return signalSemaphores[frameNumber];
}

const VkRenderPass& GraphicsLinker::getRenderPass() const {
    return renderPass;
}

const VkCommandBuffer& GraphicsLinker::getCommandBuffer(uint32_t frameNumber) const {
    return commandBuffers[frameNumber];
}

void GraphicsLinker::updateCmdFlags(){
    std::fill(updateCommandBufferFlags.begin(), updateCommandBufferFlags.end(), true);
}

}
