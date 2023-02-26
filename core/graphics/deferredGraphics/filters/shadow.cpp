#include "shadow.h"
#include "core/operations.h"
#include "core/transformational/lightInterface.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"

shadowGraphics::shadowGraphics()
{}

void shadowGraphics::setEmptyTexture(texture* emptyTexture){
    this->emptyTexture = emptyTexture;
}

void shadowGraphics::setExternalPath(const std::string& path){
    shadow.ExternalPath = path;
}

void shadowGraphics::setImageProp(imageInfo* pInfo){
    this->image = *pInfo;
}

void shadowGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device){
    this->physicalDevice = physicalDevice;
    this->device = device;
}

void shadowGraphics::setAttachments(uint32_t attachmentsCount, attachments* pAttachments){
    static_cast<void>(attachmentsCount);
    static_cast<void>(pAttachments);
}

void shadowGraphics::createAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    static_cast<void>(attachmentsCount);
    pAttachments->createDepth(physicalDevice,device,image.Format,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,image.Extent,image.Count);
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
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->sampler);
}

void shadowGraphics::Shadow::Destroy(VkDevice* device)
{
    if(Pipeline)                    {vkDestroyPipeline(*device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)              {vkDestroyPipelineLayout(*device, PipelineLayout,nullptr); PipelineLayout = VK_NULL_HANDLE;}
    if(lightUniformBufferSetLayout) {vkDestroyDescriptorSetLayout(*device, lightUniformBufferSetLayout, nullptr); lightUniformBufferSetLayout = VK_NULL_HANDLE;}
    if(uniformBlockSetLayout)       {vkDestroyDescriptorSetLayout(*device, uniformBlockSetLayout, nullptr); uniformBlockSetLayout = VK_NULL_HANDLE;}
    if(uniformBufferSetLayout)      {vkDestroyDescriptorSetLayout(*device, uniformBufferSetLayout, nullptr); uniformBufferSetLayout = VK_NULL_HANDLE;}
}

void shadowGraphics::destroy()
{
    shadow.Destroy(device);

    if(renderPass){ vkDestroyRenderPass(*device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
}

void shadowGraphics::createRenderPass()
{
    VkAttachmentDescription attachments = attachments::depthDescription(VK_FORMAT_D32_SFLOAT);

    VkAttachmentReference depthRef{};
        depthRef.attachment = 0;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pDepthStencilAttachment = &depthRef;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.pNext = NULL;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 0;
        renderPassInfo.pDependencies = NULL;
        renderPassInfo.flags = 0;
    vkCreateRenderPass(*device, &renderPassInfo, NULL, &renderPass);
}

void shadowGraphics::createPipelines()
{
    shadow.createDescriptorSetLayout(device);
    shadow.createPipeline(device,&image,&renderPass);
}

void shadowGraphics::Shadow::createDescriptorSetLayout(VkDevice* device)
{
    SpotLight::createShadowDescriptorSetLayout(*device, &lightUniformBufferSetLayout);
    object::createDescriptorSetLayout(*device, &uniformBufferSetLayout);
    gltfModel::createNodeDescriptorSetLayout(*device, &uniformBlockSetLayout);
}

void shadowGraphics::Shadow::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    auto vertShaderCode = ShaderModule::readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\shadow\\shad.spv");
    VkShaderModule vertShaderModule = ShaderModule::create(device,vertShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo};

    auto bindingDescription = gltfModel::Vertex::getShadowBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getShadowAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) pInfo->Extent.width;
    viewport.height = (float) pInfo->Extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent.width = pInfo->Extent.width;
    scissor.extent.height = pInfo->Extent.height;
    scissor.offset.x = 0;
    scissor.offset.y = 0;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 4.0f;
    rasterizer.depthBiasSlopeFactor = 1.5f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDescriptorSetLayout,3> SetLayouts = {lightUniformBufferSetLayout,uniformBufferSetLayout,uniformBlockSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

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

    VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = nullptr;
        pipelineInfo.stageCount = 1;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = PipelineLayout;
        pipelineInfo.renderPass = *pRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;
    vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline);

    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void shadowGraphics::bindBaseObject(object *newObject)
{
    shadow.objects.push_back(newObject);
}

bool shadowGraphics::removeBaseObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<shadow.objects.size(); index++){
        if(object==shadow.objects[index]){
            shadow.objects.erase(shadow.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

void shadowGraphics::createFramebuffers(light* lightSource)
{
    framebuffers[lightSource].resize(image.Count);
    for (size_t j = 0; j < image.Count; j++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &lightSource->getAttachments()->imageView[j];
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[lightSource][j]);
    }
}

void shadowGraphics::addLightSource(light* lightSource)
{
    shadow.lightSources.push_back(lightSource);
}

void shadowGraphics::removeLightSource(light* lightSource)
{
    for(uint32_t index = 0; index<shadow.lightSources.size(); index++){
        if(lightSource==shadow.lightSources[index]){

            for(uint32_t i=0;i<image.Count;i++){
                if(framebuffers[shadow.lightSources[index]][i]){
                    vkDestroyFramebuffer(*device, framebuffers[shadow.lightSources[index]][i],nullptr);
                    framebuffers[shadow.lightSources[index]][i] = VK_NULL_HANDLE;
                }
            }
            framebuffers.erase(shadow.lightSources[index]);
            shadow.lightSources.erase(shadow.lightSources.begin()+index);
            break;
        }
    }
}

void shadowGraphics::createCommandBuffers(VkCommandPool commandPool)
{
    commandBuffers.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(image.Count);
    vkAllocateCommandBuffers(*device, &allocInfo, commandBuffers.data());
}

void shadowGraphics::updateCommandBuffer(uint32_t frameNumber)
{
    vkResetCommandBuffer(commandBuffers[frameNumber],0);

     VkCommandBufferBeginInfo beginInfo{};
         beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
         beginInfo.flags = 0;
         beginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo);
        for(uint32_t attachmentNumber = 0; attachmentNumber < shadow.lightSources.size(); attachmentNumber++){
            render(frameNumber,commandBuffers[frameNumber],attachmentNumber);
        }
    vkEndCommandBuffer(commandBuffers[frameNumber]);
}

VkCommandBuffer& shadowGraphics::getCommandBuffer(uint32_t frameNumber)
{
    return commandBuffers[frameNumber];
}

void shadowGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber)
{
    std::array<VkClearValue, 1> ClearValues{};
    for(uint32_t index = 0; index < ClearValues.size(); index++)
        ClearValues[index].color = shadow.lightSources[attachmentNumber]->getAttachments()->clearValue.color;

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[shadow.lightSources[attachmentNumber]][frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.Pipeline);
        for(auto object: shadow.objects)
        {
            if(object->getEnable()&&object->getEnableShadow()){
                VkDeviceSize offsets[1] = { 0 };
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);
                if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                    vkCmdBindIndexBuffer(commandBuffer,  object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

                for (auto node : object->getModel(frameNumber)->nodes){
                    std::vector<VkDescriptorSet> descriptorSets = {shadow.lightSources[attachmentNumber]->getShadowDescriptorSets()[frameNumber],object->getDescriptorSet()[frameNumber]};
                    renderNode(commandBuffer,node,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data());
                }
            }
        }

    vkCmdEndRenderPass(commandBuffer);
}

void shadowGraphics::renderNode(VkCommandBuffer commandBuffer, Node* node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets)
{
    if (node->mesh)
    {
        std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount+1);
        for(uint32_t i=0;i<descriptorSetsCount;i++)
            nodeDescriptorSets[i] = descriptorSets[i];
        nodeDescriptorSets[descriptorSetsCount] = node->mesh->uniformBuffer.descriptorSet;

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.PipelineLayout, 0, descriptorSetsCount+1, nodeDescriptorSets.data(), 0, NULL);

        for (Primitive* primitive : node->mesh->primitives)
            if (primitive->hasIndices)  vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else                        vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
    }
    for (auto child : node->children)
        renderNode(commandBuffer, child, descriptorSetsCount, descriptorSets);
}
