#include "node.h"
#include "operations.h"

namespace moon::utils {

Stage::Stage(std::vector<VkCommandBuffer> commandBuffers, VkPipelineStageFlags waitStage, VkQueue queue) :
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

Node::Node(VkDevice device, const std::vector<Stage>& stages, Node* next) :
    stages(stages), next(next), device(device)
{}

Node::~Node(){
    if(next) delete next; next = nullptr;
    signalSemaphores.clear();
}

void Node::swap(Node& other) {
    std::swap(stages, other.stages);
    std::swap(signalSemaphores, other.signalSemaphores);
    std::swap(next, other.next);
    std::swap(device, other.device);
}

Node::Node(Node&& other) {
    swap(other);
}

Node& Node::operator=(Node&& other) {
    swap(other);
    return *this;
}

Node* Node::back() {
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

VkResult Node::createSemaphores(){
    VkResult result = VK_SUCCESS;

    auto createSemaphore = [this](VkDevice device, Stage* stage){
        auto& signalSemaphore = signalSemaphores.emplace_back();
        signalSemaphore = utils::vkDefault::Semaphore(device);
        stage->signalSemaphores.push_back(signalSemaphore);
        return VK_SUCCESS;
    };

    if(next){
        for(auto& currentStage: stages){
            for(auto& nextStage: next->stages){
                result = std::max(result, createSemaphore(device, &currentStage));
                nextStage.waitSemaphores.push_back(signalSemaphores.back());
            }
        }
        next->createSemaphores();
    }else{
        for(auto& currentStage: stages){
            result = std::max(result, createSemaphore(device, &currentStage));
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
