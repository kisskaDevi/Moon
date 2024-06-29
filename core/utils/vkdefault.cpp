#include "vkdefault.h"
#include "operations.h"

#include <glfw3.h>

namespace moon::utils {

VkSamplerCreateInfo vkDefault::sampler(){
    VkSamplerCreateInfo SamplerInfo{};
        SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        SamplerInfo.magFilter = VK_FILTER_LINEAR;
        SamplerInfo.minFilter = VK_FILTER_LINEAR;
        SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.anisotropyEnable = VK_TRUE;
        SamplerInfo.maxAnisotropy = 1.0f;
        SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        SamplerInfo.unnormalizedCoordinates = VK_FALSE;
        SamplerInfo.compareEnable = VK_FALSE;
        SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        SamplerInfo.minLod = 0.0f;
        SamplerInfo.maxLod = 0.0f;
        SamplerInfo.mipLodBias = 0.0f;
    return SamplerInfo;
}

VkPipelineInputAssemblyStateCreateInfo vkDefault::inputAssembly(){
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
    return inputAssembly;
}

VkViewport vkDefault::viewport(VkOffset2D offset, VkExtent2D extent){
    VkViewport viewport{};
        viewport.x = static_cast<float>(offset.x);
        viewport.y = static_cast<float>(offset.y);
        viewport.width  = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
    return viewport;
}

VkRect2D vkDefault::scissor(VkOffset2D offset, VkExtent2D extent){
    VkRect2D scissor{};
        scissor.offset = offset;
        scissor.extent = extent;
    return scissor;
}

VkPipelineViewportStateCreateInfo vkDefault::viewportState(VkViewport* viewport, VkRect2D* scissor){
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = scissor;
    return viewportState;
}

VkPipelineRasterizationStateCreateInfo vkDefault::rasterizationState(){
    VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
    return rasterizer;
}

VkPipelineRasterizationStateCreateInfo vkDefault::rasterizationState(VkFrontFace frontFace){
    VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = frontFace;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
    return rasterizer;
}

VkPipelineMultisampleStateCreateInfo vkDefault::multisampleState(){
    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;
    return multisampling;
}

VkPipelineDepthStencilStateCreateInfo vkDefault::depthStencilDisable(){
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};
    return depthStencil;
}

VkPipelineDepthStencilStateCreateInfo vkDefault::depthStencilEnable(){
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};
    return depthStencil;
}

VkPipelineColorBlendAttachmentState vkDefault::colorBlendAttachmentState(VkBool32 enable){
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = enable;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_MAX;
    return colorBlendAttachment;
}

VkPipelineColorBlendStateCreateInfo vkDefault::colorBlendState(uint32_t attachmentCount, VkPipelineColorBlendAttachmentState* pAttachments){
    VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = attachmentCount;
        colorBlending.pAttachments = pAttachments;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;
    return colorBlending;
}

VkPipelineVertexInputStateCreateInfo vkDefault::vertexInputState(){
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    return vertexInputInfo;
}

VkDescriptorSetLayoutBinding vkDefault::bufferVertexLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::bufferFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::imageFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::inAttachmentFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

#define VKDEFAULT_MAKE_DESCRIPTOR(Name, BaseDescriptor)                                 \
	vkDefault::Name::~Name() { destroy();}				                                \
	vkDefault::Name::Name(vkDefault::Name&& other) noexcept {                           \
        swap(other);                                                                    \
    }                                                                                   \
	vkDefault::Name& vkDefault::Name::operator=(vkDefault::Name&& other) noexcept {     \
        swap(other);                                                                    \
        return *this;                                                                   \
    }                                                                                   \
	vkDefault::Name::operator const BaseDescriptor&() const {                           \
        return descriptor;                                                              \
    }                                                                                   \
	vkDefault::Name::operator const BaseDescriptor*() const {                           \
        return &descriptor;                                                             \
    }

#define VKDEFAULT_MAKE_SWAP(Name)				                                        \
    void vkDefault::Name::swap(vkDefault::Name& other) noexcept {                       \
        uint8_t buff[sizeof(Name)];                                                     \
        std::memcpy((void*)buff, (void*)&other, sizeof(Name));                          \
        std::memcpy((void*)&other, (void*)this, sizeof(Name));                          \
        std::memcpy((void*)this, (void*)buff, sizeof(Name));                            \
    }

#define VKDEFAULT_RESET()				                                                \
    destroy();                                                                          \
    this->device = device;                                                              \

template<typename Descriptor>
Descriptor release(Descriptor& descriptor) {
    Descriptor temp = descriptor;
    descriptor = VK_NULL_HANDLE;
    return temp;
}

VkResult vkDefault::Pipeline::create(VkDevice device, const std::vector<VkGraphicsPipelineCreateInfo>& graphicsPipelineCreateInfos) {
    VKDEFAULT_RESET()

    return vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(graphicsPipelineCreateInfos.size()), graphicsPipelineCreateInfos.data(), nullptr, &descriptor);
}

void vkDefault::Pipeline::destroy() {
    if (descriptor) vkDestroyPipeline(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Pipeline)
VKDEFAULT_MAKE_DESCRIPTOR(Pipeline, VkPipeline)

VkResult vkDefault::PipelineLayout::create(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
    const std::vector<VkPushConstantRange>& pushConstantRange)
{
    VKDEFAULT_RESET()

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
        pipelineLayoutCreateInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRange.data();

    return vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &descriptor);
}

void vkDefault::PipelineLayout::destroy() {
    if (descriptor) vkDestroyPipelineLayout(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(PipelineLayout)
VKDEFAULT_MAKE_DESCRIPTOR(PipelineLayout, VkPipelineLayout)

VkResult vkDefault::DescriptorSetLayout::create(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VKDEFAULT_RESET()

    this->bindings = bindings;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    return vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptor);
}

void vkDefault::DescriptorSetLayout::destroy() {
    bindings.clear();
    if (descriptor) vkDestroyDescriptorSetLayout(device, release(descriptor), nullptr);
}

void vkDefault::DescriptorSetLayout::swap(DescriptorSetLayout& other) noexcept {
    std::swap(bindings, other.bindings);
    std::swap(descriptor, other.descriptor);
    std::swap(device, other.device);
}

VKDEFAULT_MAKE_DESCRIPTOR(DescriptorSetLayout, VkDescriptorSetLayout)

vkDefault::ShaderModule::~ShaderModule() {
    destroy();
}

vkDefault::ShaderModule::ShaderModule(VkDevice device, const std::filesystem::path& shaderPath) :
    shaderModule(moon::utils::shaderModule::create(device, moon::utils::shaderModule::readFile(shaderPath))), device(device)
{}

void vkDefault::ShaderModule::destroy() {
    if (shaderModule) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }
}

vkDefault::ShaderModule::operator const VkShaderModule& () const {
    return shaderModule;
}

vkDefault::FragmentShaderModule::~FragmentShaderModule() {
    destroy();
}

void vkDefault::FragmentShaderModule::destroy() {
    vkDefault::ShaderModule::destroy();
    pipelineShaderStageCreateInfo = VkPipelineShaderStageCreateInfo{};
    specializationInfo = VkSpecializationInfo{};
}

vkDefault::FragmentShaderModule::FragmentShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo) :
    ShaderModule(device, shaderPath), specializationInfo(specializationInfo) {
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    pipelineShaderStageCreateInfo.module = shaderModule;
    pipelineShaderStageCreateInfo.pName = "main";
    pipelineShaderStageCreateInfo.pSpecializationInfo = &this->specializationInfo;
}

vkDefault::FragmentShaderModule::operator const VkPipelineShaderStageCreateInfo& () const {
    return pipelineShaderStageCreateInfo;
}

vkDefault::VertrxShaderModule::~VertrxShaderModule() {
    destroy();
}

void vkDefault::VertrxShaderModule::destroy() {
    vkDefault::ShaderModule::destroy();
    pipelineShaderStageCreateInfo = VkPipelineShaderStageCreateInfo{};
    specializationInfo = VkSpecializationInfo{};
}

vkDefault::VertrxShaderModule::VertrxShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo) :
    ShaderModule(device, shaderPath), specializationInfo(specializationInfo) {
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    pipelineShaderStageCreateInfo.module = shaderModule;
    pipelineShaderStageCreateInfo.pName = "main";
    pipelineShaderStageCreateInfo.pSpecializationInfo = &this->specializationInfo;
}

vkDefault::VertrxShaderModule::operator const VkPipelineShaderStageCreateInfo& () const {
    return pipelineShaderStageCreateInfo;
}

VkResult vkDefault::RenderPass::create(VkDevice device, const AttachmentDescriptions& attachments, const SubpassDescriptions& subpasses, const SubpassDependencies& dependencies) {
    VKDEFAULT_RESET()

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
        renderPassInfo.pSubpasses = subpasses.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

    return vkCreateRenderPass(device, &renderPassInfo, nullptr, &descriptor);
}

void vkDefault::RenderPass::destroy() {
    if (descriptor) vkDestroyRenderPass(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(RenderPass)
VKDEFAULT_MAKE_DESCRIPTOR(RenderPass, VkRenderPass)

VkResult vkDefault::Framebuffer::create(VkDevice device, const VkFramebufferCreateInfo& framebufferInfo) {
    VKDEFAULT_RESET()

    return vkCreateFramebuffer(device, &framebufferInfo, nullptr, &descriptor);
}

void vkDefault::Framebuffer::destroy() {
    if (descriptor) vkDestroyFramebuffer(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Framebuffer)
VKDEFAULT_MAKE_DESCRIPTOR(Framebuffer, VkFramebuffer)

vkDefault::Instance::~Instance() {
    destroy();
}

VkResult vkDefault::Instance::create(const VkInstanceCreateInfo& createInfo) {
    return vkCreateInstance(&createInfo, nullptr, &instance);
}

void vkDefault::Instance::destroy() {
    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

vkDefault::Instance::operator const VkInstance& () const {
    return instance;
}

vkDefault::DebugUtilsMessenger::~DebugUtilsMessenger() {
    destroy();
}

void vkDefault::DebugUtilsMessenger::create(const VkInstance& instance) {
    destroy();
    this->instance = instance;
    moon::utils::validationLayer::setupDebugMessenger(instance, &debugUtilsMessenger);
}

void vkDefault::DebugUtilsMessenger::destroy() {
    if (debugUtilsMessenger) {
        moon::utils::validationLayer::destroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
        debugUtilsMessenger = VK_NULL_HANDLE;
    }
}

vkDefault::DebugUtilsMessenger::operator const VkDebugUtilsMessengerEXT& () const {
    return debugUtilsMessenger;
}

vkDefault::Surface::~Surface() {
    destroy();
}

VkResult vkDefault::Surface::create(const VkInstance& instance, GLFWwindow* window) {
    destroy();
    this->instance = instance;
    return glfwCreateWindowSurface(instance, window, nullptr, &surface);
}

void vkDefault::Surface::destroy() {
    if (surface) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }
}

vkDefault::Surface::operator const VkSurfaceKHR& () const {
    return surface;
}

VkResult vkDefault::Semaphore::create(const VkDevice& device) {
    VKDEFAULT_RESET()

    VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    return vkCreateSemaphore(device, &semaphoreInfo, nullptr, &descriptor);
}

void vkDefault::Semaphore::destroy() {
    if(descriptor) vkDestroySemaphore(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Semaphore)
VKDEFAULT_MAKE_DESCRIPTOR(Semaphore, VkSemaphore)

VkResult vkDefault::Fence::create(const VkDevice& device) {
    VKDEFAULT_RESET()

    VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    return vkCreateFence(device, &fenceInfo, nullptr, &descriptor);
}

void vkDefault::Fence::destroy() {
    if (descriptor) vkDestroyFence(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Fence)
VKDEFAULT_MAKE_DESCRIPTOR(Fence, VkFence)

VkResult vkDefault::Sampler::create(const VkDevice& device, const VkSamplerCreateInfo& samplerInfo) {
    VKDEFAULT_RESET()

    return vkCreateSampler(device, &samplerInfo, nullptr, &descriptor);
}

void vkDefault::Sampler::destroy() {
    if (descriptor) vkDestroySampler(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Sampler)
VKDEFAULT_MAKE_DESCRIPTOR(Sampler, VkSampler)

VkResult vkDefault::DescriptorPool::create(const VkDevice& device, const VkDescriptorPoolCreateInfo& poolInfo) {
    VKDEFAULT_RESET()
    return vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptor);
}

VkResult vkDefault::DescriptorPool::create(const VkDevice& device, const std::vector<const vkDefault::DescriptorSetLayout*>& descriptorSetLayouts, const uint32_t descriptorsCount) {
    uint32_t maxSets = 0;
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const vkDefault::DescriptorSetLayout* descriptorSetLayout : descriptorSetLayouts) {
        for (const VkDescriptorSetLayoutBinding& binding : descriptorSetLayout->bindings) {
            VkDescriptorPoolSize descriptorPoolSize;
                descriptorPoolSize.type = binding.descriptorType;
                descriptorPoolSize.descriptorCount = static_cast<uint32_t>(binding.descriptorCount * descriptorsCount);
                maxSets += descriptorPoolSize.descriptorCount;
            poolSizes.push_back(descriptorPoolSize);
        }
    }
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = maxSets;
    return create(device, poolInfo);
}

void vkDefault::DescriptorPool::destroy() {
    if (descriptor) vkDestroyDescriptorPool(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(DescriptorPool)
VKDEFAULT_MAKE_DESCRIPTOR(DescriptorPool, VkDescriptorPool)

VkResult vkDefault::ImageView::create(
    const VkDevice& device,
    const VkImage& image,
    VkImageViewType type,
    VkFormat format,
    VkImageAspectFlags aspectFlags,
    uint32_t mipLevels,
    uint32_t baseArrayLayer,
    uint32_t layerCount) {
    VKDEFAULT_RESET()
    return utils::texture::createView(device, type, format, aspectFlags, mipLevels, baseArrayLayer, layerCount, image, &descriptor);
}

void vkDefault::ImageView::destroy() {
    if (descriptor) vkDestroyImageView(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(ImageView)
VKDEFAULT_MAKE_DESCRIPTOR(ImageView, VkImageView)

VkResult vkDefault::Image::create(
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
    VkMemoryPropertyFlags           properties) {
    VKDEFAULT_RESET()
    return utils::texture::create(physicalDevice, device, flags, extent, arrayLayers, mipLevels, numSamples, format, layout, usage, properties, &descriptor, &memory);
}

void vkDefault::Image::destroy() {
    utils::texture::destroy(device, descriptor, memory);
}

vkDefault::Image::operator const VkDeviceMemory& () const {
    return memory;
}

vkDefault::Image::operator const VkDeviceMemory* () const {
    return &memory;
}

VKDEFAULT_MAKE_SWAP(Image)
VKDEFAULT_MAKE_DESCRIPTOR(Image, VkImage)

void vkDefault::Buffer::destroy() {
    utils::buffer::destroy(device, descriptor, memory);
}

VkResult vkDefault::Buffer::create(
    VkPhysicalDevice                physicalDevice,
    VkDevice                        device,
    VkDeviceSize                    size,
    VkBufferUsageFlags              usage,
    VkMemoryPropertyFlags           properties) {
    VKDEFAULT_RESET()
    memorysize = size;
    VkResult result = utils::buffer::create(physicalDevice, device, size, usage, properties, &descriptor, &memory);
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT || properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        result = std::max(result, vkMapMemory(device, memory, 0, size, 0, &memorymap));
    }
    return result;
}

void vkDefault::Buffer::copy(const void* data) {
    std::memcpy(memorymap, data, memorysize);
}

void vkDefault::Buffer::copy(const void* data, uint32_t offset, uint32_t size) {
    if (memorymap) {
        vkUnmapMemory(device, memory);
    }
    CHECK(vkMapMemory(device, memory, offset, size, 0, &memorymap));
    std::memcpy(memorymap, data, size);
}

size_t vkDefault::Buffer::size() const {
    return memorysize;
}

void vkDefault::Buffer::raiseFlag() {
    updateFlag = true;
}

bool vkDefault::Buffer::dropFlag() {
    bool temp = updateFlag;
    updateFlag = false;
    return temp;
}

void* &vkDefault::Buffer::map() {
    return memorymap;
}

vkDefault::Buffer::operator const VkDeviceMemory& () const {
    return memory;
}

vkDefault::Buffer::operator const VkDeviceMemory* () const {
    return &memory;
}

VKDEFAULT_MAKE_SWAP(Buffer)
VKDEFAULT_MAKE_DESCRIPTOR(Buffer, VkBuffer)
}
