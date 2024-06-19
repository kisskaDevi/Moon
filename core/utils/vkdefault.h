#ifndef VKDEFAULT_H
#define VKDEFAULT_H

#include <vulkan.h>
#include <vector>
#include <filesystem>
#include <unordered_map>

struct GLFWwindow;

namespace moon::utils::vkDefault {

VkSamplerCreateInfo sampler();

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
	void destroy();
public:
	~Pipeline();
	Pipeline() = default;
	Pipeline(const Pipeline&) = delete;
	Pipeline& operator=(const Pipeline&) = delete;
	Pipeline(Pipeline&& other) {
		std::swap(pipeline, other.pipeline);
		std::swap(device, other.device);
	}
	Pipeline& operator=(Pipeline&& other) {
		std::swap(pipeline, other.pipeline);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(VkDevice device, const std::vector<VkGraphicsPipelineCreateInfo>& graphicsPipelineCreateInfos);
	operator const VkPipeline&() const;
};

using PipelineMap = std::unordered_map<MaskType, moon::utils::vkDefault::Pipeline>;

class PipelineLayout {
private:
	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	void destroy();
public:
	~PipelineLayout();
	PipelineLayout() = default;
	PipelineLayout(const PipelineLayout&) = delete;
	PipelineLayout& operator=(const PipelineLayout&) = delete;
	PipelineLayout(PipelineLayout&& other) {
		std::swap(pipelineLayout, other.pipelineLayout);
		std::swap(device, other.device);
	}
	PipelineLayout& operator=(PipelineLayout&& other) {
		std::swap(pipelineLayout, other.pipelineLayout);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(
		VkDevice device,
		const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
		const std::vector<VkPushConstantRange>& pushConstantRange = {});
	operator const VkPipelineLayout& () const;
};

using PipelineLayoutMap = std::unordered_map<MaskType, moon::utils::vkDefault::PipelineLayout>;

class DescriptorSetLayout {
private:
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	void destroy();
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
		bindings = std::move(other.bindings);
		std::swap(descriptorSetLayout, other.descriptorSetLayout);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings);
	operator const VkDescriptorSetLayout&() const;
	operator const VkDescriptorSetLayout*() const;
};

using DescriptorSetLayoutMap = std::unordered_map<MaskType, moon::utils::vkDefault::DescriptorSetLayout>;

class ShaderModule {
protected:
	VkShaderModule shaderModule{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	virtual void destroy();
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
	operator const VkShaderModule& () const;
};

class FragmentShaderModule : public ShaderModule {
private:
	VkSpecializationInfo specializationInfo{};
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{};
	void destroy();
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
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		std::swap(specializationInfo, other.specializationInfo);
		return *this;
	}

	FragmentShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo = VkSpecializationInfo{});
	operator const VkPipelineShaderStageCreateInfo& () const;
};

class VertrxShaderModule : public ShaderModule {
private:
	VkSpecializationInfo specializationInfo{};
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{};
	void destroy();
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
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		std::swap(specializationInfo, other.specializationInfo);
		return *this;
	}

	VertrxShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo = VkSpecializationInfo{});
	operator const VkPipelineShaderStageCreateInfo& () const;
};

class RenderPass {
private:
	VkRenderPass renderPass{VK_NULL_HANDLE};
	VkDevice device{ VK_NULL_HANDLE };
	void destroy();
public:
	~RenderPass();
	RenderPass() = default;
	RenderPass(const RenderPass&) = delete;
	RenderPass& operator=(const RenderPass&) = delete;
	RenderPass(RenderPass&& other) {
		std::swap(renderPass, other.renderPass);
		std::swap(device, other.device);
	}
	RenderPass& operator=(RenderPass&& other) {
		std::swap(renderPass, other.renderPass);
		std::swap(device, other.device);
		return *this;
	}

	using AttachmentDescriptions = std::vector<VkAttachmentDescription>;
	using SubpassDescriptions = std::vector<VkSubpassDescription>;
	using SubpassDependencies = std::vector<VkSubpassDependency>;

	VkResult create(
		VkDevice device,
		const AttachmentDescriptions& attachments,
		const SubpassDescriptions& subpasses,
		const SubpassDependencies& dependencies);
	operator const VkRenderPass& () const;
};

class Framebuffer {
private:
	VkFramebuffer framebuffer{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	void destroy();

public:
	~Framebuffer();
	Framebuffer() = default;
	Framebuffer(const Framebuffer&) = delete;
	Framebuffer& operator=(const Framebuffer&) = delete;
	Framebuffer(Framebuffer&& other) {
		std::swap(framebuffer, other.framebuffer);
		std::swap(device, other.device);
	}
	Framebuffer& operator=(Framebuffer&& other) {
		std::swap(framebuffer, other.framebuffer);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(VkDevice device, const VkFramebufferCreateInfo& framebufferInfo);
	operator const VkFramebuffer& () const;
};

using Framebuffers = std::vector<Framebuffer>;

class Instance {
private:
	VkInstance instance{ VK_NULL_HANDLE };
	void destroy();
public:
	~Instance();
	Instance() = default;
	Instance(const Instance&) = delete;
	Instance& operator=(const Instance&) = delete;
	Instance(Instance&& other) {
		std::swap(instance, other.instance);
	}
	Instance& operator=(Instance&& other) {
		std::swap(instance, other.instance);
		return *this;
	}

	VkResult create(const VkInstanceCreateInfo& createInfo);
	operator const VkInstance& () const;
};

class DebugUtilsMessenger {
private:
	VkDebugUtilsMessengerEXT debugUtilsMessenger{ VK_NULL_HANDLE };
	VkInstance instance{ VK_NULL_HANDLE };
	void destroy();
public:
	~DebugUtilsMessenger();
	DebugUtilsMessenger() = default;
	DebugUtilsMessenger(const DebugUtilsMessenger&) = delete;
	DebugUtilsMessenger& operator=(const DebugUtilsMessenger&) = delete;
	DebugUtilsMessenger(DebugUtilsMessenger&& other) {
		std::swap(debugUtilsMessenger, other.debugUtilsMessenger);
		std::swap(instance, other.instance);
	}
	DebugUtilsMessenger& operator=(DebugUtilsMessenger&& other) {
		std::swap(debugUtilsMessenger, other.debugUtilsMessenger);
		std::swap(instance, other.instance);
		return *this;
	}

	void create(const VkInstance& createInfo);
	operator const VkDebugUtilsMessengerEXT& () const;
};

class Surface {
private:
	VkSurfaceKHR surface{ VK_NULL_HANDLE };
	VkInstance instance{ VK_NULL_HANDLE };
	void destroy();
public:
	~Surface();
	Surface() = default;
	Surface(const Surface&) = delete;
	Surface& operator=(const Surface&) = delete;
	Surface(Surface&& other) {
		std::swap(surface, other.surface);
		std::swap(instance, other.instance);
	}
	Surface& operator=(Surface&& other) {
		std::swap(surface, other.surface);
		std::swap(instance, other.instance);
		return *this;
	}

	VkResult create(const VkInstance& instance, GLFWwindow* window);
	operator const VkSurfaceKHR& () const;
};

class Semaphore {
private:
	VkSemaphore semaphore{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	void destroy();
public:
	~Semaphore();
	Semaphore() = default;
	Semaphore(const Semaphore&) = delete;
	Semaphore& operator=(const Semaphore&) = delete;
	Semaphore(Semaphore&& other) {
		std::swap(semaphore, other.semaphore);
		std::swap(device, other.device);
	}
	Semaphore& operator=(Semaphore&& other) {
		std::swap(semaphore, other.semaphore);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(const VkDevice& device);
	operator const VkSemaphore& () const;
	operator const VkSemaphore* () const;
};

using Semaphores = std::vector<Semaphore>;

class Fence {
private:
	VkFence fence{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	void destroy();
public:
	~Fence();
	Fence() = default;
	Fence(const Fence&) = delete;
	Fence& operator=(const Fence&) = delete;
	Fence(Fence&& other) {
		std::swap(fence, other.fence);
		std::swap(device, other.device);
	}
	Fence& operator=(Fence&& other) {
		std::swap(fence, other.fence);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(const VkDevice& device);
	operator const VkFence& () const;
	operator const VkFence* () const;
};

using Fences = std::vector<Fence>;

class Sampler {
private:
	VkSampler sampler{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	void destroy();
public:
	~Sampler();
	Sampler() = default;
	Sampler(const Sampler&) = delete;
	Sampler& operator=(const Sampler&) = delete;
	Sampler(Sampler&& other) {
		std::swap(sampler, other.sampler);
		std::swap(device, other.device);
	}
	Sampler& operator=(Sampler&& other) {
		std::swap(sampler, other.sampler);
		std::swap(device, other.device);
		return *this;
	}

	VkResult create(const VkDevice& device, const VkSamplerCreateInfo& samplerInfo);
	operator const VkSampler& () const;
};

}
#endif // VKDEFAULT_H
