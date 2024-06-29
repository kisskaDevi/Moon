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

#define VKDEFAULT_INIT_DESCRIPTOR(Name, BaseDescriptor)	\
private:												\
	BaseDescriptor descriptor{ VK_NULL_HANDLE };		\
	VkDevice device{ VK_NULL_HANDLE };					\
	void destroy();										\
public:													\
	~Name();											\
	Name() noexcept = default;							\
	Name(const Name& other) = delete;					\
	Name& operator=(const Name& other) = delete;		\
	Name(Name&& other) noexcept;						\
	Name& operator=(Name&& other) noexcept;				\
	void swap(Name& other) noexcept;					\
	operator const BaseDescriptor&() const;				\
	operator const BaseDescriptor*() const;

using MaskType = uint8_t;

class Pipeline {
	VKDEFAULT_INIT_DESCRIPTOR(Pipeline, VkPipeline)

	VkResult create(VkDevice device, const std::vector<VkGraphicsPipelineCreateInfo>& graphicsPipelineCreateInfos);
};

using PipelineMap = std::unordered_map<MaskType, moon::utils::vkDefault::Pipeline>;

class PipelineLayout {
	VKDEFAULT_INIT_DESCRIPTOR(PipelineLayout, VkPipelineLayout)

	VkResult create(
		VkDevice device,
		const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
		const std::vector<VkPushConstantRange>& pushConstantRange = {});
};

using PipelineLayoutMap = std::unordered_map<MaskType, moon::utils::vkDefault::PipelineLayout>;

class DescriptorSetLayout {
public:
	std::vector<VkDescriptorSetLayoutBinding> bindings;

	VKDEFAULT_INIT_DESCRIPTOR(DescriptorSetLayout, VkDescriptorSetLayout)

	VkResult create(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings);
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
	ShaderModule(ShaderModule&& other) noexcept { swap(other);}
	ShaderModule& operator=(ShaderModule&& other) noexcept { swap(other); return *this;}
	void swap(ShaderModule& other) noexcept {
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
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
	FragmentShaderModule(FragmentShaderModule&& other) noexcept { swap(other); }
	FragmentShaderModule& operator=(FragmentShaderModule&& other) noexcept { swap(other); return *this; }
	void swap(FragmentShaderModule& other) noexcept {
		std::swap(specializationInfo, other.specializationInfo);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		ShaderModule::swap(other);
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
	VertrxShaderModule(VertrxShaderModule&& other) noexcept { swap(other); }
	VertrxShaderModule& operator=(VertrxShaderModule&& other) noexcept { swap(other); return *this; }
	void swap(VertrxShaderModule& other) noexcept {
		std::swap(specializationInfo, other.specializationInfo);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		ShaderModule::swap(other);
	}

	VertrxShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo = VkSpecializationInfo{});
	operator const VkPipelineShaderStageCreateInfo& () const;
};

class RenderPass {
	VKDEFAULT_INIT_DESCRIPTOR(RenderPass, VkRenderPass)

	using AttachmentDescriptions = std::vector<VkAttachmentDescription>;
	using SubpassDescriptions = std::vector<VkSubpassDescription>;
	using SubpassDependencies = std::vector<VkSubpassDependency>;

	VkResult create(VkDevice device, const AttachmentDescriptions& attachments, const SubpassDescriptions& subpasses, const SubpassDependencies& dependencies);
};

class Framebuffer {
	VKDEFAULT_INIT_DESCRIPTOR(Framebuffer, VkFramebuffer)

	VkResult create(VkDevice device, const VkFramebufferCreateInfo& framebufferInfo);
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
	Instance(Instance&& other) noexcept { std::swap(instance, other.instance); }
	Instance& operator=(Instance&& other) noexcept { std::swap(instance, other.instance); return *this; }

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
	DebugUtilsMessenger(DebugUtilsMessenger&& other) noexcept { swap(other); }
	DebugUtilsMessenger& operator=(DebugUtilsMessenger&& other) noexcept { swap(other); return *this; }
	void swap(DebugUtilsMessenger& other) noexcept {
		std::swap(debugUtilsMessenger, other.debugUtilsMessenger);
		std::swap(instance, other.instance);
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
	Surface(Surface&& other) { swap(other); }
	Surface& operator=(Surface&& other) { swap(other); return *this; }
	void swap(Surface& other) {
		std::swap(surface, other.surface);
		std::swap(instance, other.instance);
	}

	VkResult create(const VkInstance& instance, GLFWwindow* window);
	operator const VkSurfaceKHR& () const;
};

class Semaphore {
private:
	VKDEFAULT_INIT_DESCRIPTOR(Semaphore, VkSemaphore)

	VkResult create(const VkDevice& device);
};

using Semaphores = std::vector<Semaphore>;

class Fence {
	VKDEFAULT_INIT_DESCRIPTOR(Fence, VkFence)

	VkResult create(const VkDevice& device);
};

using Fences = std::vector<Fence>;

class Sampler {
	VKDEFAULT_INIT_DESCRIPTOR(Sampler, VkSampler)

	VkResult create(const VkDevice& device, const VkSamplerCreateInfo& samplerInfo);
};

class DescriptorPool {
	VKDEFAULT_INIT_DESCRIPTOR(DescriptorPool, VkDescriptorPool)

	VkResult create(const VkDevice& device, const std::vector<const vkDefault::DescriptorSetLayout*>& descriptorSetLayouts, const uint32_t descriptorsCount);
	VkResult create(const VkDevice& device, const VkDescriptorPoolCreateInfo& poolInfo);
};

class ImageView {
	VKDEFAULT_INIT_DESCRIPTOR(ImageView, VkImageView)

	VkResult create(
		const VkDevice& device,
		const VkImage& image,
		VkImageViewType type,
		VkFormat format,
		VkImageAspectFlags aspectFlags,
		uint32_t mipLevels,
		uint32_t baseArrayLayer,
		uint32_t layerCount);
};

class Image {
private:
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	VKDEFAULT_INIT_DESCRIPTOR(Image, VkImage)

	VkResult create(
		VkPhysicalDevice                physicalDevice,
		VkDevice                        device,
		VkImageCreateFlags              flags,
		VkExtent3D                      extent,
		uint32_t                        arrayLayers,
		uint32_t                        mipLevels,
		VkSampleCountFlagBits           numSamples,
		VkFormat                        format,
		VkImageLayout                   layout,
		VkImageUsageFlags               usage,
		VkMemoryPropertyFlags           properties);

	operator const VkDeviceMemory& () const;
	operator const VkDeviceMemory* () const;
};

}
#endif // VKDEFAULT_H
