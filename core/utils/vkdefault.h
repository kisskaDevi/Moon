#ifndef VKDEFAULT_H
#define VKDEFAULT_H

#include <vulkan.h>
#include <vector>
#include <filesystem>
#include <unordered_map>

namespace moon::utils::vkDefault {

VkSamplerCreateInfo samler();

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

class ShaderModule {
protected:
	VkShaderModule shaderModule{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
public:
	virtual ~ShaderModule();
	ShaderModule() = default;
	ShaderModule(const ShaderModule&) = delete;
	ShaderModule& operator=(const ShaderModule&) = delete;
	ShaderModule(ShaderModule&& other) {
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
	}
	ShaderModule& operator=(ShaderModule&& other) {
		destroy();
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
		return *this;
	}

	ShaderModule(VkDevice device, const std::filesystem::path& shaderPath);
	virtual void destroy();
	operator const VkShaderModule& () const;
};

class FragmentShaderModule : public ShaderModule {
private:
	VkSpecializationInfo specializationInfo{};
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{};
public:
	~FragmentShaderModule();
	FragmentShaderModule() = default;
	FragmentShaderModule(const FragmentShaderModule&) = delete;
	FragmentShaderModule& operator=(const FragmentShaderModule&) = delete;
	FragmentShaderModule(FragmentShaderModule&& other) {
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		std::swap(specializationInfo, other.specializationInfo);
	}
	FragmentShaderModule& operator=(FragmentShaderModule&& other) {
		destroy();
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		std::swap(specializationInfo, other.specializationInfo);
		return *this;
	}

	FragmentShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo = VkSpecializationInfo{});
	void destroy();
	operator const VkPipelineShaderStageCreateInfo& () const;
};

class VertrxShaderModule : public ShaderModule {
private:
	VkSpecializationInfo specializationInfo{};
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{};
public:
	~VertrxShaderModule();
	VertrxShaderModule() = default;
	VertrxShaderModule(const VertrxShaderModule&) = delete;
	VertrxShaderModule& operator=(const VertrxShaderModule&) = delete;
	VertrxShaderModule(VertrxShaderModule&& other) {
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		std::swap(specializationInfo, other.specializationInfo);
	}
	VertrxShaderModule& operator=(VertrxShaderModule&& other) {
		destroy();
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		std::swap(specializationInfo, other.specializationInfo);
		return *this;
	}

	VertrxShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo = VkSpecializationInfo{});
	void destroy();
	operator const VkPipelineShaderStageCreateInfo& () const;
};

}
#endif // VKDEFAULT_H
