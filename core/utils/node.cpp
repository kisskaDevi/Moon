#include "node.h"
#include "operations.h"

#include <unordered_set>

namespace moon::utils {

PipelineStage::PipelineStage(const std::vector<const vkDefault::CommandBuffers*>& commandBuffers, VkPipelineStageFlags waitStagesMask, VkQueue queue)
    : waitStagesMask(waitStagesMask), queue(queue)
{
    std::unordered_set<size_t> sizes;
    for (const auto& frameCommandBuffers : commandBuffers) {
        sizes.insert(frameCommandBuffers->size());
    }

    auto sizeCheck = CHECK_M(sizes.size() != 1, std::string("[PipelineStage::PipelineStage] input commandBuffers must be same size"))
    if (sizeCheck) return;

    frames.resize(*sizes.begin());
    for (const auto& frameCommandBuffers : commandBuffers) {
        for (const auto& commandBuffers : *frameCommandBuffers) {
            const auto frameIndex = &commandBuffers - &frameCommandBuffers->front();
            frames.at(frameIndex).commandBuffers.push_back(commandBuffers);
        }
    }
}

VkResult PipelineStage::submit(uint32_t frameIndex) const {
    const auto& frame = frames.at(frameIndex);
    std::vector<VkPipelineStageFlags> waitStagesMasks(frame.wait.size(), waitStagesMask);
    std::vector<VkSemaphore> signals(frame.signal.begin(), frame.signal.end());
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(frame.wait.size());
        submitInfo.pWaitSemaphores = frame.wait.data();
        submitInfo.pWaitDstStageMask = waitStagesMasks.data();
        submitInfo.commandBufferCount = static_cast<uint32_t>(frame.commandBuffers.size());
        submitInfo.pCommandBuffers = frame.commandBuffers.data();
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signals.size());
        submitInfo.pSignalSemaphores = signals.data();
    return vkQueueSubmit(queue, 1, &submitInfo, fence);
}

PipelineNode::PipelineNode(VkDevice device, PipelineStages&& instages, PipelineNode* next) : stages(std::move(instages)), next(next)
{
    for (auto& currentStage : stages) {
        for (auto& frame : currentStage.frames) {
            if (next) {
                const auto frameIndex = &frame - &currentStage.frames.front();
                for (auto& nextStage : next->stages) {
                    auto& signal = frame.signal.emplace_back(utils::vkDefault::Semaphore(device));
                    nextStage.frames.at(frameIndex).wait.push_back(signal);
                }
            } else {
                frame.signal.emplace_back(utils::vkDefault::Semaphore(device));
            }
        }
    }
}

std::vector<std::vector<VkSemaphore>> PipelineNode::submit(const uint32_t frameIndex, const std::vector<VkFence>& externalFence, const std::vector<std::vector<VkSemaphore>>& externalSemaphore){
    for (uint32_t i = 0; i < externalSemaphore.size(); i++) {
        stages[i].frames.at(frameIndex).wait = externalSemaphore[i];
    }

    if (!next) {
        for (uint32_t i = 0; i < externalFence.size(); i++) {
            stages[i].fence = externalFence[i];
        }
    }

    for(const auto& stage: stages){
        stage.submit(frameIndex);
    }

    return next ? next->submit(frameIndex, externalFence) : semaphores(frameIndex);
}

std::vector<std::vector<VkSemaphore>> PipelineNode::semaphores(uint32_t frameIndex) {
    std::vector<std::vector<VkSemaphore>> semaphores;
    for (const auto& stage : stages) {
        const auto& frame = stage.frames.at(frameIndex);
        std::vector<VkSemaphore> signals(frame.signal.begin(), frame.signal.end());
        semaphores.push_back(signals);
    }
    return semaphores;
}

}
