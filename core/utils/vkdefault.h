#ifndef VKDEFAULT_H
#define VKDEFAULT_H

#include <vulkan.h>
#include <vector>
#include <unordered_map>

namespace moon::utils::vkDefault {

VkSamplerCreateInfo samler();

VkPipelineShaderStageCreateInfo fragmentShaderStage(VkShaderModule shaderModule);
VkPipelineShaderStageCreateInfo vertrxShaderStage(VkShaderModule shaderModule);

VkPipelineVertexInputStateCreateInfo vertexInputState();
VkViewport viewport(VkOffset2D offset, VkExtent2D extent);
VkRect2D scissor(VkOffset2D offset, VkExtent2D extent);
VkPipelineViewportStateCreateInfo viewportState(VkViewport* viewport, VkRect2D* scissor);
VkPipelineInputAssemblyStateCreateInfo inputAssembly();
VkPipelineRasterizationStateCreateInfo rasterizationState();
VkPipelineRasterizationStateCreateInfo rasterizationState(VkFrontFace frontFace);
VkPipelineMultisampleStateCreateInfo multisampleState();
VkPipelineDepthStencilStateCreateInfo depthStencilDisable();
VkPipelineDepthStencilStateCreateInfo depthStencilEnable();
VkPipelineColorBlendAttachmentState colorBlendAttachmentState(VkBool32 enable);
VkPipelineColorBlendStateCreateInfo colorBlendState(uint32_t attachmentCount, VkPipelineColorBlendAttachmentState* pAttachments);

VkDescriptorSetLayoutBinding bufferVertexLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding bufferFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding imageFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding inAttachmentFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);

using MaskType = uint8_t;

class Pipeline {
private:
	VkPipeline pipeline{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
public:
	~Pipeline();
	Pipeline() = default;
	Pipeline(const Pipeline&) = delete;
	Pipeline& operator=(const Pipeline&) = delete;
	Pipeline(Pipeline&& other) {
		std::swap(pipeline, other.pipeline);
	}
	Pipeline& operator=(Pipeline&& other) {
		destroy();
		std::swap(pipeline, other.pipeline);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(VkDevice device, const std::vector<VkGraphicsPipelineCreateInfo>& graphicsPipelineCreateInfos);
	void destroy();
	operator const VkPipeline&() const;
};

using PipelineMap = std::unordered_map<MaskType, moon::utils::vkDefault::Pipeline>;

class PipelineLayout {
private:
	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
public:
	~PipelineLayout();
	PipelineLayout() = default;
	PipelineLayout(const PipelineLayout&) = delete;
	PipelineLayout& operator=(const PipelineLayout&) = delete;
	PipelineLayout(PipelineLayout&& other) {
		std::swap(pipelineLayout, other.pipelineLayout);
	}
	PipelineLayout& operator=(PipelineLayout&& other) {
		destroy();
		std::swap(pipelineLayout, other.pipelineLayout);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(
		VkDevice device,
		const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
		const std::vector<VkPushConstantRange>& pushConstantRange = {});
	void destroy();
	operator const VkPipelineLayout& () const;
};

using PipelineLayoutMap = std::unordered_map<MaskType, moon::utils::vkDefault::PipelineLayout>;

class DescriptorSetLayout {
private:
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
public:
	~DescriptorSetLayout();
	DescriptorSetLayout() = default;
	DescriptorSetLayout(const DescriptorSetLayout&) = delete;
	DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;
	DescriptorSetLayout(DescriptorSetLayout&& other) : bindings(std::move(other.bindings)) {
		std::swap(descriptorSetLayout, other.descriptorSetLayout);
		std::swap(device, other.device);
	}
	DescriptorSetLayout& operator=(DescriptorSetLayout&& other) {
		destroy();
		bindings = std::move(other.bindings);
		std::swap(descriptorSetLayout, other.descriptorSetLayout);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings);
	void destroy();
	operator const VkDescriptorSetLayout&() const;
	operator const VkDescriptorSetLayout*() const;
};

using DescriptorSetLayoutMap = std::unordered_map<MaskType, moon::utils::vkDefault::DescriptorSetLayout>;

}
#endif // VKDEFAULT_H
