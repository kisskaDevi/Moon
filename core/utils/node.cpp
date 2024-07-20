#include "node.h"
#include "operations.h"

namespace moon::utils {

PipelineStage::PipelineStage(std::vector<VkCommandBuffer> commandBuffers, VkPipelineStageFlags waitStage, VkQueue queue) :
    commandBuffers(commandBuffers),
    waitStage(waitStage),
    queue(queue)
{}

VkResult PipelineStage::submit(){
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

PipelineNode::PipelineNode(VkDevice device, const PipelineStages& stages, PipelineNode* next) :
    stages(stages), next(next), device(device)
{}

PipelineNode::~PipelineNode(){
    if(next) delete next; next = nullptr;
    signalSemaphores.clear();
}

void PipelineNode::swap(PipelineNode& other) {
    std::swap(stages, other.stages);
    std::swap(signalSemaphores, other.signalSemaphores);
    std::swap(next, other.next);
    std::swap(device, other.device);
}

PipelineNode::PipelineNode(PipelineNode&& other) {
    swap(other);
}

PipelineNode& PipelineNode::operator=(PipelineNode&& other) {
    swap(other);
    return *this;
}

PipelineNode* PipelineNode::back() {
    return next ? next->back() : this;
}

void PipelineNode::setExternalSemaphore(const std::vector<std::vector<VkSemaphore>>& externalSemaphore){
    for(uint32_t i = 0; i < externalSemaphore.size(); i++){
        stages[i].waitSemaphores = externalSemaphore[i];
    }
}

void PipelineNode::setExternalFence(const std::vector<VkFence>& externalFence){
    for(uint32_t i = 0; i < externalFence.size(); i++){
        stages[i].fence = externalFence[i];
    }
}

std::vector<std::vector<VkSemaphore>> PipelineNode::getBackSemaphores(){
    std::vector<std::vector<VkSemaphore>> semaphores;
    for(const auto& stage: stages){
        semaphores.push_back(stage.signalSemaphores);
    }
    return semaphores;
}

VkResult PipelineNode::createSemaphores(){
    auto createSemaphore = [this](VkDevice device, PipelineStage* stage){
        auto& signalSemaphore = signalSemaphores.emplace_back();
        signalSemaphore = utils::vkDefault::Semaphore(device);
        stage->signalSemaphores.push_back(signalSemaphore);
    };

    if(next){
        for(auto& currentStage: stages){
            for(auto& nextStage: next->stages){
                createSemaphore(device, &currentStage);
                nextStage.waitSemaphores.push_back(signalSemaphores.back());
            }
        }
        next->createSemaphores();
    }else{
        for(auto& currentStage: stages){
            createSemaphore(device, &currentStage);
        }
    }
    return VK_SUCCESS;
}

void PipelineNode::submit(){
    for(auto& stage: stages){
        stage.submit();
    }
    if(next){
        next->submit();
    }
}

}
