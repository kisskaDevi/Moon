#include "graphicsLinker.h"
#include "attachments.h"
#include "swapChain.h"
#include "linkable.h"

graphicsLinker::~graphicsLinker(){
    destroy();
}

void graphicsLinker::destroy(){
    if(renderPass) {
        vkDestroyRenderPass(device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;
    }

    for(auto& framebuffer: framebuffers){
        if(framebuffer) vkDestroyFramebuffer(device, framebuffer,nullptr);
    }
    framebuffers.clear();

    if(!commandBuffers.empty()){
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    }
    commandBuffers.clear();

    if(commandPool) {
        vkDestroyCommandPool(device, commandPool, nullptr); commandPool = VK_NULL_HANDLE;
    }

    for (auto& signalSemaphore: signalSemaphores){
        if(signalSemaphore) vkDestroySemaphore(device, signalSemaphore, nullptr);
    }
    signalSemaphores.clear();

    updateCommandBufferFlags.clear();
}

void graphicsLinker::setSwapChain(swapChain* swapChainKHR){
    this->swapChainKHR = swapChainKHR;
    this->imageCount = swapChainKHR->getImageCount();
    this->imageExtent = swapChainKHR->getExtent();
    this->imageFormat = swapChainKHR->getFormat();
}

void graphicsLinker::setDevice(VkDevice device){
    this->device = device;
}

void graphicsLinker::addLinkable(linkable* link){
    linkables.push_back(link);
    updateCmdFlags();
}

void graphicsLinker::createRenderPass(){
    std::vector<VkAttachmentDescription> attachments = {
        attachments::imageDescription(imageFormat, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
        attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    std::vector<VkSubpassDescription> subpass;
    for(auto refIt = attachmentRef.begin(); refIt != attachmentRef.end(); refIt++){
        subpass.push_back(VkSubpassDescription{});
            subpass.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
            subpass.back().pColorAttachments = refIt->data();
    }

    std::vector<VkSubpassDependency> dependency;
    dependency.push_back(VkSubpassDependency{});
        dependency.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.back().dstSubpass = 0;
        dependency.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependency.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependency.size());
        renderPassInfo.pDependencies = dependency.data();
    vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
}

void graphicsLinker::createFramebuffers(){
    framebuffers.resize(imageCount);
    for (size_t Image = 0; Image < framebuffers.size(); Image++)
    {
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &swapChainKHR->attachment(Image).imageView;
            framebufferInfo.width = imageExtent.width;
            framebufferInfo.height = imageExtent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[Image]);
    }
}

void graphicsLinker::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
}

void graphicsLinker::createCommandBuffers(){
    commandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(imageCount);
    vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());

    updateCommandBufferFlags.resize(imageCount, true);
}

void graphicsLinker::updateCommandBuffer(uint32_t resourceNumber, uint32_t imageNumber){
    if(updateCommandBufferFlags[resourceNumber])
    {
        vkResetCommandBuffer(commandBuffers[resourceNumber],0);

        VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;
        vkBeginCommandBuffer(commandBuffers[resourceNumber], &beginInfo);

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

        vkEndCommandBuffer(commandBuffers[resourceNumber]);

        updateCommandBufferFlags[resourceNumber] = false;
    }
}

void graphicsLinker::createSyncObjects(){
    signalSemaphores.resize(imageCount);

    for (auto& signalSemaphore: signalSemaphores){
        VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &signalSemaphore);
    }
}

const VkSemaphore& graphicsLinker::submit(uint32_t frameNumber, const std::vector<VkSemaphore>& waitSemaphores, VkFence fence, VkQueue queue){
    VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = submitInfo.waitSemaphoreCount > 0 ? waitSemaphores.data() : VK_NULL_HANDLE;
        submitInfo.pWaitDstStageMask = &waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[frameNumber];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &signalSemaphores[frameNumber];
    vkQueueSubmit(queue, 1, &submitInfo, fence);
    return signalSemaphores[frameNumber];
}

const VkRenderPass& graphicsLinker::getRenderPass() const {
    return renderPass;
}

const VkCommandBuffer& graphicsLinker::getCommandBuffer(uint32_t frameNumber) const {
    return commandBuffers[frameNumber];
}

void graphicsLinker::updateCmdFlags(){
    std::fill(updateCommandBufferFlags.begin(), updateCommandBufferFlags.end(), true);
}
