#include "vkdefault.h"
#include "operations.h"

#include <glfw3.h>

namespace moon::utils {

VkSamplerCreateInfo vkDefault::samler(){
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

vkDefault::Pipeline::~Pipeline() {
    destroy();
}

VkResult vkDefault::Pipeline::create(VkDevice device, const std::vector<VkGraphicsPipelineCreateInfo>& graphicsPipelineCreateInfos) {
    destroy();
    this->device = device;
    return vkCreateGraphicsPipelines(
        device,
        VK_NULL_HANDLE,
        static_cast<uint32_t>(graphicsPipelineCreateInfos.size()),
        graphicsPipelineCreateInfos.data(),
        nullptr,
        &pipeline);
}

void vkDefault::Pipeline::destroy() {
    if (pipeline) {
        vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
}

vkDefault::Pipeline::operator const VkPipeline&() const {
    return pipeline;
}

vkDefault::PipelineLayout::~PipelineLayout() {
    destroy();
}

VkResult vkDefault::PipelineLayout::create(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
    const std::vector<VkPushConstantRange>& pushConstantRange)
{
    destroy();
    this->device = device;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
        pipelineLayoutCreateInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRange.data();

    return vkCreatePipelineLayout(
        device,
        &pipelineLayoutCreateInfo,
        nullptr,
        &pipelineLayout);
}

void vkDefault::PipelineLayout::destroy() {
    if (pipelineLayout) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
}

vkDefault::PipelineLayout::operator const VkPipelineLayout&() const {
    return pipelineLayout;
}

vkDefault::DescriptorSetLayout::~DescriptorSetLayout() {
    destroy();
}

VkResult vkDefault::DescriptorSetLayout::create(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    destroy();
    this->device = device;
    this->bindings = bindings;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    return vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
}

void vkDefault::DescriptorSetLayout::destroy() {
    bindings.clear();
    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
}

vkDefault::DescriptorSetLayout::operator const VkDescriptorSetLayout& () const {
    return descriptorSetLayout;
}

vkDefault::DescriptorSetLayout::operator const VkDescriptorSetLayout* () const {
    return &descriptorSetLayout;
}

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

vkDefault::RenderPass::~RenderPass() {
    destroy();
}

VkResult vkDefault::RenderPass::create(
    VkDevice device,
    const AttachmentDescriptions& attachments,
    const SubpassDescriptions& subpasses,
    const SubpassDependencies& dependencies) {
    destroy();
    this->device = device;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
        renderPassInfo.pSubpasses = subpasses.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

    return vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
}

void vkDefault::RenderPass::destroy() {
    if (renderPass) {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }
}

vkDefault::RenderPass::operator const VkRenderPass& () const {
    return renderPass;
}

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

vkDefault::Semaphore::~Semaphore() {
    destroy();
}

VkResult vkDefault::Semaphore::create(const VkDevice& device) {
    destroy();
    this->device = device;
    VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    return vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore);
}

void vkDefault::Semaphore::destroy() {
    if(semaphore){
        vkDestroySemaphore(device, semaphore, nullptr);
        semaphore = VK_NULL_HANDLE;
    }
}

vkDefault::Semaphore::operator const VkSemaphore& () const {
    return semaphore;
}

vkDefault::Semaphore::operator const VkSemaphore* () const {
    return &semaphore;
}

vkDefault::Fence::~Fence() {
    destroy();
}

VkResult vkDefault::Fence::create(const VkDevice& device) {
    destroy();
    this->device = device;
    VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    return vkCreateFence(device, &fenceInfo, nullptr, &fence);
}

void vkDefault::Fence::destroy() {
    if (fence) {
        vkDestroyFence(device, fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
}

vkDefault::Fence::operator const VkFence& () const {
    return fence;
}

vkDefault::Fence::operator const VkFence* () const {
    return &fence;
}

}
