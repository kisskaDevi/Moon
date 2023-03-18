#include "node.h"

stage::stage(   std::vector<VkCommandBuffer> commandBuffers,
                std::vector<VkPipelineStageFlags> waitStages,
                VkQueue queue) :
    commandBuffers(commandBuffers),
    waitStages(waitStages),
    queue(queue)
{}

VkResult stage::submit(){
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = waitSemaphores.size();
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = commandBuffers.size();
        submitInfo.pCommandBuffers = commandBuffers.data();
        submitInfo.signalSemaphoreCount = signalSemaphores.size();
        submitInfo.pSignalSemaphores = signalSemaphores.data();
    return vkQueueSubmit(queue, 1, &submitInfo, fence);
}

node::node(const std::vector<stage>& stages, node* next) :
    stages(stages), next(next)
{}

void node::destroy(VkDevice device){
    if(next){
        next->destroy(device); delete next; next = nullptr;
    }
    for(auto& semaphore: signalSemaphores){
        vkDestroySemaphore(device, semaphore, nullptr);
    }
}

node* node::back(){
    return next ? next->back() : this;
}

void node::setExternalSemaphore(const std::vector<std::vector<VkSemaphore>>& externalSemaphore){
    for(uint32_t i = 0; i < externalSemaphore.size(); i++){
        stages[i].waitSemaphores = externalSemaphore[i];
    }
}

void node::setExternalFence(std::vector<VkFence>& externalFence){
    for(uint32_t i = 0; i < externalFence.size(); i++){
        stages[i].fence = externalFence[i];
    }
}

std::vector<std::vector<VkSemaphore>> node::getBackSemaphores(){
    std::vector<std::vector<VkSemaphore>> semaphores;
    for(const auto& stage: stages){
        semaphores.push_back(stage.signalSemaphores);
    }
    return semaphores;
}

void node::createSemaphores(VkDevice device){
    auto createSemaphore = [this](VkDevice device, VkSemaphoreCreateInfo* semaphoreInfo, stage* stage){
        signalSemaphores.push_back(VkSemaphore{});
        vkCreateSemaphore(device, semaphoreInfo, nullptr, &signalSemaphores.back());
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
}

void node::submit(){
    for(auto& stage: stages){
        stage.submit();
    }
    if(next){
        next->submit();
    }
}