#include "shadowGraphics.h"
#include "core/transformational/object.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"

#include <array>
#include <iostream>

shadowGraphics::shadowGraphics(uint32_t imageCount, VkExtent2D shadowExtent)
{
    image.Count = imageCount;
    image.Extent.width = shadowExtent.width;
    image.Extent.height = shadowExtent.height;
    image.Samples = VK_SAMPLE_COUNT_1_BIT;
}

void shadowGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->queueFamilyIndices = queueFamilyIndices;
    image.Format = findDepthFormat(this->physicalDevice);
}

void shadowGraphics::createAttachments()
{
    createImage(    physicalDevice,
                    device,
                    image.Extent.width,
                    image.Extent.height,
                    1.0f,
                    VK_SAMPLE_COUNT_1_BIT,
                    image.Format,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthAttachment.image,
                    depthAttachment.imageMemory);
    depthAttachment.imageView =
            createImageView(    device,
                                depthAttachment.image,
                                image.Format,
                                VK_IMAGE_ASPECT_DEPTH_BIT,
                                1.0f);
    VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.minLod = static_cast<float>(1.0f);
        samplerInfo.maxLod = static_cast<float>(1.0f);
        samplerInfo.mipLodBias = 0.0f;
    vkCreateSampler(*device, &samplerInfo, nullptr, &depthAttachment.sampler);
}

void shadowGraphics::Shadow::Destroy(VkDevice* device)
{
    if(Pipeline)                vkDestroyPipeline(*device, Pipeline, nullptr);
    if(PipelineLayout)          vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    if(DescriptorSetLayout)     vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    if(uniformBlockSetLayout)   vkDestroyDescriptorSetLayout(*device, uniformBlockSetLayout, nullptr);
    if(uniformBufferSetLayout)  vkDestroyDescriptorSetLayout(*device, uniformBufferSetLayout, nullptr);
    if(DescriptorPool)          vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
}

void shadowGraphics::destroy()
{
    shadow.Destroy(device);

    if(RenderPass) vkDestroyRenderPass(*device, RenderPass, nullptr);
    for(uint32_t i=0;i<shadowMapFramebuffer.size();i++)
        if(shadowMapFramebuffer[i]) vkDestroyFramebuffer(*device, shadowMapFramebuffer[i],nullptr);

    if(shadowCommandBuffer.data())  vkFreeCommandBuffers(*device, shadowCommandPool, static_cast<uint32_t>(shadowCommandBuffer.size()), shadowCommandBuffer.data());
    if(shadowCommandPool)           vkDestroyCommandPool(*device, shadowCommandPool, nullptr);

    depthAttachment.deleteAttachment(device);
    depthAttachment.deleteSampler(device);
}

void shadowGraphics::setExternalPath(const std::string &path)
{
    shadow.ExternalPath = path;
}

void shadowGraphics::createRenderPass()
{
    VkAttachmentDescription attachments{};
        attachments.format =  findDepthFormat(physicalDevice);
        attachments.samples = VK_SAMPLE_COUNT_1_BIT;
        attachments.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.flags = 0;

    VkAttachmentReference depthRef{};
        depthRef.attachment = 0;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass;
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.flags = 0;
        subpass.inputAttachmentCount = 0;
        subpass.pInputAttachments = NULL;
        subpass.colorAttachmentCount = 0;
        subpass.pColorAttachments = NULL;
        subpass.pResolveAttachments = NULL;
        subpass.pDepthStencilAttachment = &depthRef;
        subpass.preserveAttachmentCount = 0;
        subpass.pPreserveAttachments = NULL;

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
    vkCreateRenderPass(*device, &renderPassInfo, NULL, &RenderPass);
}

void shadowGraphics::createFramebuffer()
{
    shadowMapFramebuffer.resize(image.Count);
    for (size_t i = 0; i < shadowMapFramebuffer.size(); i++)
    {
        VkFramebufferCreateInfo framebufferInfo;
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.pNext = NULL;
            framebufferInfo.renderPass = RenderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &depthAttachment.imageView;
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
            framebufferInfo.flags = 0;
        vkCreateFramebuffer(*device, &framebufferInfo, NULL, &shadowMapFramebuffer[i]);
    }
}

void shadowGraphics::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices->graphicsFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(*device, &poolInfo, nullptr, &shadowCommandPool);
}

void shadowGraphics::Shadow::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\shadow\\shad.spv");
    VkShaderModule vertShaderModule = createShaderModule(device,vertShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo};

    auto bindingDescription = gltfModel::Vertex::getShadowBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getShadowAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();                            //указатель на массив соответствующийх структуру

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                        //без него кажда полоса и веер потребуют отдельной команды вывода.
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
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;                             //параметр направления обхода (против часовой стрелки)
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
    colorBlending.logicOpEnable = VK_FALSE;                                         //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                       //Optional
    colorBlending.attachmentCount = 1;                                              //количество подключений
    colorBlending.pAttachments = &colorBlendAttachment;                             //массив подключений

    std::array<VkDescriptorSetLayout,3> SetLayouts = {DescriptorSetLayout,uniformBufferSetLayout,uniformBlockSetLayout};
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
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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

void shadowGraphics::Shadow::createDescriptorSetLayout(VkDevice* device)
{
    VkDescriptorSetLayoutBinding lightUboLayoutBinding={};
        lightUboLayoutBinding.binding = 0;
        lightUboLayoutBinding.descriptorCount = 1;
        lightUboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lightUboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        lightUboLayoutBinding.pImmutableSamplers = nullptr;
    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {lightUboLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout);

    VkDescriptorSetLayoutBinding uniformBufferLayoutBinding{};
        uniformBufferLayoutBinding.binding = 0;
        uniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBufferLayoutBinding.descriptorCount = 1;
        uniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniformBufferLayoutBinding.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = 1;
        uniformBufferLayoutInfo.pBindings = &uniformBufferLayoutBinding;
    vkCreateDescriptorSetLayout(*device, &uniformBufferLayoutInfo, nullptr, &uniformBufferSetLayout);

    VkDescriptorSetLayoutBinding uniformBlockLayoutBinding{};
        uniformBlockLayoutBinding.binding = 0;
        uniformBlockLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBlockLayoutBinding.descriptorCount = 1;
        uniformBlockLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniformBlockLayoutBinding.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo uniformBlockLayoutInfo{};
        uniformBlockLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBlockLayoutInfo.bindingCount = 1;
        uniformBlockLayoutInfo.pBindings = &uniformBlockLayoutBinding;
    vkCreateDescriptorSetLayout(*device, &uniformBlockLayoutInfo, nullptr, &uniformBlockSetLayout);
}

void shadowGraphics::createDescriptorPool()
{
    size_t index = 0;
    std::vector<VkDescriptorPoolSize> poolSizes(1);
        poolSizes[index].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[index].descriptorCount = static_cast<uint32_t>(image.Count);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &shadow.DescriptorPool);
}

void shadowGraphics::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(image.Count, shadow.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = shadow.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    shadow.DescriptorSets.resize(image.Count);
    vkAllocateDescriptorSets(*device, &allocInfo, shadow.DescriptorSets.data());
}

void shadowGraphics::updateDescriptorSets(uint32_t lightUniformBuffersCount, VkBuffer* plightUniformBuffers, unsigned long long sizeOfLightUniformBuffers)
{
    for (size_t i=0; i<lightUniformBuffersCount; i++)
    {
        VkDescriptorBufferInfo lightBufferInfo;
            lightBufferInfo.buffer = plightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeOfLightUniformBuffers;
        VkWriteDescriptorSet descriptorWrites{};
            descriptorWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.dstSet = shadow.DescriptorSets[i];
            descriptorWrites.dstBinding = 0;
            descriptorWrites.dstArrayElement = 0;
            descriptorWrites.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.descriptorCount = 1;
            descriptorWrites.pBufferInfo = &lightBufferInfo;
        vkUpdateDescriptorSets(*device, 1, &descriptorWrites, 0, nullptr);
    }
}

void shadowGraphics::createCommandBuffers()
{
    shadowCommandBuffer.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = shadowCommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = image.Count;
    vkAllocateCommandBuffers(*device, &allocInfo, shadowCommandBuffer.data());
}

void shadowGraphics::updateCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects)
{
    VkClearValue clearValues{};
        clearValues.depthStencil.depth = 1.0f;
        clearValues.depthStencil.stencil = 0;

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;
    vkBeginCommandBuffer(shadowCommandBuffer[frameNumber], &beginInfo);

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.pNext = NULL;
        renderPassInfo.renderPass = RenderPass;
        renderPassInfo.framebuffer = shadowMapFramebuffer[frameNumber];
        renderPassInfo.renderArea.offset.x = 0;
        renderPassInfo.renderArea.offset.y = 0;
        renderPassInfo.renderArea.extent.width = image.Extent.width;
        renderPassInfo.renderArea.extent.height = image.Extent.height;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearValues;

    vkCmdBeginRenderPass(shadowCommandBuffer[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport;
            viewport.width = image.Extent.width;
            viewport.height = image.Extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            viewport.x = 0;
            viewport.y = 0;
        vkCmdSetViewport(shadowCommandBuffer[frameNumber], 0, 1, &viewport);

        VkRect2D scissor;
            scissor.extent.width = image.Extent.width;
            scissor.extent.height = image.Extent.height;
            scissor.offset.x = 0;
            scissor.offset.y = 0;
        vkCmdSetScissor(shadowCommandBuffer[frameNumber], 0, 1, &scissor);

        vkCmdBindPipeline(shadowCommandBuffer[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.Pipeline);
        for(auto object: objects)
        {
            if(object->getEnable()&&object->getEnableShadow()){
                VkDeviceSize offsets[1] = { 0 };
                vkCmdBindVertexBuffers(shadowCommandBuffer[frameNumber], 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);
                if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                    vkCmdBindIndexBuffer(shadowCommandBuffer[frameNumber],  object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

                for (auto node : object->getModel(frameNumber)->nodes){
                    std::vector<VkDescriptorSet> descriptorSets = {shadow.DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                    renderNode(shadowCommandBuffer[frameNumber],node,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data());
                }
            }
        }

    vkCmdEndRenderPass(shadowCommandBuffer[frameNumber]);

    vkEndCommandBuffer(shadowCommandBuffer[frameNumber]);
}

void shadowGraphics::renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets)
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

void shadowGraphics::createShadow()
{
    createCommandPool();
    createAttachments();
    createRenderPass();
    createFramebuffer();
    shadow.createDescriptorSetLayout(device);
    shadow.createPipeline(device,&image,&RenderPass);
    createDescriptorPool();
    createDescriptorSets();
}

attachment*                     shadowGraphics::getAttachment(){return &depthAttachment;}
VkCommandBuffer*                shadowGraphics::getCommandBuffer(uint32_t i){return &shadowCommandBuffer[i];}
