#include "workflow.h"
#include "operations.h"

namespace moon::workflows {

Workflow& Workflow::setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device){
    this->physicalDevice = physicalDevice;
    this->device = device;
    return *this;
}

void Workflow::update(uint32_t frameNumber) {
    if (commandBuffers[frameNumber].dropFlag()) {
        CHECK(commandBuffers[frameNumber].reset());
        CHECK(commandBuffers[frameNumber].begin());
        updateCommandBuffer(frameNumber);
        CHECK(commandBuffers[frameNumber].end());
    }
}

utils::vkDefault::CommandBuffer& Workflow::commandBuffer(uint32_t frameNumber) {
    return commandBuffers[frameNumber];
}

void Workflow::raiseUpdateFlags() {
    utils::raiseFlags(commandBuffers);
}

}
