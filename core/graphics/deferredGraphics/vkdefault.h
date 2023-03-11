#ifndef VKDEFAULT_H
#define VKDEFAULT_H

#include <vulkan/vulkan.h>

namespace vkDefault {

VkSamplerCreateInfo samler();

VkPipelineShaderStageCreateInfo fragmentShaderStage(VkShaderModule shaderModule);
VkPipelineShaderStageCreateInfo vertrxShaderStage(VkShaderModule shaderModule);

VkPipelineVertexInputStateCreateInfo vertexInputState();
VkViewport viewport(VkExtent2D extent);
VkRect2D scissor(VkExtent2D extent);
VkPipelineViewportStateCreateInfo viewportState(VkViewport* viewport, VkRect2D* scissor);
VkPipelineInputAssemblyStateCreateInfo inputAssembly();
VkPipelineRasterizationStateCreateInfo rasterizationState();
VkPipelineMultisampleStateCreateInfo multisampleState();
VkPipelineDepthStencilStateCreateInfo depthStencilDisable();
VkPipelineDepthStencilStateCreateInfo depthStencilEnable();
VkPipelineColorBlendAttachmentState colorBlendAttachmentState(VkBool32 enable);
VkPipelineColorBlendStateCreateInfo colorBlendState(uint32_t attachmentCount, VkPipelineColorBlendAttachmentState* pAttachments);

VkDescriptorSetLayoutBinding bufferVertexLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding bufferFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding imageFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);
}

#endif // VKDEFAULT_H
