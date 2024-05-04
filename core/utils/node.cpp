#include "node.h"
#include "operations.h"

namespace moon::utils {

Stage::Stage(   std::vector<VkCommandBuffer> commandBuffers,
                VkPipelineStageFlags waitStage,
                VkQueue queue) :
    commandBuffers(commandBuffers),
    waitStage(waitStage),
    queue(queue)
{}

VkResult Stage::submit(){
    std::vector<VkPipelineStageFlags> waitStages(waitSemaphores.size(), waitStage);
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
        submitInfo.pCommandBuffers = commandBuffers.data();
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
        submitInfo.pSignalSemaphores = signalSemaphores.data();
    return vkQueueSubmit(queue, 1, &submitInfo, fence);
}

Node::Node(const std::vector<Stage>& stages, Node* next) :
    stages(stages), next(next)
{}

void Node::destroy(VkDevice device){
    if(next){
        next->destroy(device); delete next; next = nullptr;
    }
    for(auto& semaphore: signalSemaphores){
        vkDestroySemaphore(device, semaphore, nullptr);
    }
}

Node* Node::back(){
    return next ? next->back() : this;
}

void Node::setExternalSemaphore(const std::vector<std::vector<VkSemaphore>>& externalSemaphore){
    for(uint32_t i = 0; i < externalSemaphore.size(); i++){
        stages[i].waitSemaphores = externalSemaphore[i];
    }
}

void Node::setExternalFence(const std::vector<VkFence>& externalFence){
    for(uint32_t i = 0; i < externalFence.size(); i++){
        stages[i].fence = externalFence[i];
    }
}

std::vector<std::vector<VkSemaphore>> Node::getBackSemaphores(){
    std::vector<std::vector<VkSemaphore>> semaphores;
    for(const auto& stage: stages){
        semaphores.push_back(stage.signalSemaphores);
    }
    return semaphores;
}

VkResult Node::createSemaphores(VkDevice device){
    VkResult result = VK_SUCCESS;
    auto createSemaphore = [this, &result](VkDevice device, VkSemaphoreCreateInfo* semaphoreInfo, Stage* stage){
        signalSemaphores.push_back(VkSemaphore{});
        result = vkCreateSemaphore(device, semaphoreInfo, nullptr, &signalSemaphores.back());
        CHECK(result);
        stage->signalSemaphores.push_back(signalSemaphores.back());
    };

    if(VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0}; next){
        for(auto& currentStage: stages){
            for(auto& nextStage: next->stages){
                createSemaphore(device, &semaphoreInfo, &currentStage);
                nextStage.waitSemaphores.push_back(signalSemaphores.back());
            }
        }
        next->createSemaphores(device);
    }else{
        for(auto& currentStage: stages){
            createSemaphore(device, &semaphoreInfo, &currentStage);
        }
    }
    return result;
}

void Node::submit(){
    for(auto& stage: stages){
        stage.submit();
    }
    if(next){
        next->submit();
    }
}

}
