#include "sslr.h"
#include "core/operations.h"
#include "../bufferObjects.h"

#include <array>
#include <iostream>

SSLRGraphics::SSLRGraphics()
{

}

void SSLRGraphics::setExternalPath(const std::string &path)
{
    sslr.ExternalPath = path;
}

void SSLRGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
}
void SSLRGraphics::setImageProp(imageInfo* pInfo)                       {this->image = *pInfo;}
void SSLRGraphics::setAttachments(uint32_t attachmentsCount, attachments* Attachments)
{
    this->attachmentsCount = attachmentsCount;
    this->Attachments = Attachments;
}

void SSLRGraphics::SSLR::Destroy(VkDevice* device)
{
    vkDestroyPipeline(*device, Pipeline, nullptr);
    vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
}

void SSLRGraphics::createAttachments(uint32_t attachmentsCount, attachments* Attachments)
{
    static_cast<void>(attachmentsCount);
    Attachments->resize(image.Count);
    for(size_t imageNumber=0; imageNumber<image.Count; imageNumber++)
    {
        createImage(            physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                image.Format,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT ,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                Attachments->image[imageNumber],
                                Attachments->imageMemory[imageNumber]);

        Attachments->imageView[imageNumber] =
        createImageView(        device,
                                Attachments->image[imageNumber],
                                image.Format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1);
    }
    VkSamplerCreateInfo ssaoSamplerInfo{};
        ssaoSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ssaoSamplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
        ssaoSamplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
        ssaoSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Режим адресации
        ssaoSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
        ssaoSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Повторение текстуры при выходе за пределы размеров изображения.
        ssaoSamplerInfo.anisotropyEnable = VK_TRUE;
        ssaoSamplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
        ssaoSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
        ssaoSamplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
        ssaoSamplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
        ssaoSamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
        ssaoSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ssaoSamplerInfo.minLod = 0.0f;
        ssaoSamplerInfo.maxLod = 0.0f;
        ssaoSamplerInfo.mipLodBias = 0.0f;
    if (vkCreateSampler(*device, &ssaoSamplerInfo, nullptr, &Attachments->sampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing sampler!");
}

void SSLRGraphics::destroy()
{
    sslr.Destroy(device);

    vkDestroyRenderPass(*device, renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        vkDestroyFramebuffer(*device, framebuffers[i],nullptr);
}

void SSLRGraphics::createRenderPass()
{
    uint32_t index = 0;
    std::array<VkAttachmentDescription,1> attachments{};
        attachments[index].format = image.Format;
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    index = 0;
    std::array<VkAttachmentReference,1> attachmentRef;
        attachmentRef[index].attachment = 0;
        attachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    index = 0;
    std::array<VkSubpassDescription,1> subpass{};
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(attachmentRef.size());
        subpass[index].pColorAttachments = attachmentRef.data();

    index = 0;
    std::array<VkSubpassDependency,1> dependency{};                                                                                        //зависимости
        dependency[index].srcSubpass = VK_SUBPASS_EXTERNAL;                                                                                //ссылка из исходного прохода (создавшего данные)
        dependency[index].dstSubpass = 0;                                                                                                  //в целевой подпроход (поглощающий данные)
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;                                                                 //задаёт как стадии конвейера в исходном проходе создают данные
        dependency[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;                                                                        //поля задают как каждый из исходных проходов обращается к данным
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependency.size());
        renderPassInfo.pDependencies = dependency.data();

    if (vkCreateRenderPass(*device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create filter render pass!");
}

void SSLRGraphics::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &Attachments->imageView[i];
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        if (vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing framebuffer!");
    }
}

void SSLRGraphics::createPipelines()
{
    sslr.createDescriptorSetLayout(device);
    sslr.createPipeline(device,&image,&renderPass);
}

void SSLRGraphics::SSLR::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;

    std::array<VkDescriptorSetLayoutBinding,4> bindings{};
        bindings[index].binding = 0;
        bindings[index].descriptorCount = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[index].pImmutableSamplers = nullptr;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
        bindings[index].binding = 1;
        bindings[index].descriptorCount = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[index].pImmutableSamplers = nullptr;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
        bindings[index].binding = 2;
        bindings[index].descriptorCount = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[index].pImmutableSamplers = nullptr;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
        bindings[index].binding = 3;
        bindings[index].descriptorCount = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[index].pImmutableSamplers = nullptr;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create sslr descriptor set layout!");
}

void SSLRGraphics::SSLR::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\sslr\\sslrVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\sslr\\sslrFrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[index].module = vertShaderModule;
        shaderStages[index].pName = "main";
    index++;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[index].module = fragShaderModule;
        shaderStages[index].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

    index = 0;
    std::array<VkViewport,1> viewport{};
        viewport[index].x = 0.0f;
        viewport[index].y = 0.0f;
        viewport[index].width  = (float) pInfo->Extent.width;
        viewport[index].height = (float) pInfo->Extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = pInfo->Extent;
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = static_cast<uint32_t>(viewport.size());;
        viewportState.pViewports = viewport.data();
        viewportState.scissorCount = static_cast<uint32_t>(scissor.size());;
        viewportState.pScissors = scissor.data();

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

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    index = 0;
    std::array<VkPipelineColorBlendAttachmentState,1> colorBlendAttachment;
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());
        colorBlending.pAttachments = colorBlendAttachment.data();
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

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

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create sslr pipeline layout!");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
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
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create sslr graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void SSLRGraphics::createDescriptorPool()
{
    uint32_t imageCount = image.Count;
    size_t index = 0;
    std::array<VkDescriptorPoolSize,4> poolSizes;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &sslr.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing descriptor pool 1!");
}

void SSLRGraphics::createDescriptorSets()
{
    sslr.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, sslr.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = sslr.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(*device, &allocInfo, sslr.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate postProcessing descriptor sets 1!");
}

void SSLRGraphics::updateDescriptorSets(DeferredAttachments Attachments, VkBuffer* pUniformBuffers)
{
    for (size_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo;
            bufferInfo.buffer = pUniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

        uint32_t index = 0;
        std::array<VkDescriptorImageInfo, 3> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments.GBuffer.position.imageView[i];
            imageInfo[index].sampler = Attachments.GBuffer.position.sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments.GBuffer.normal.imageView[i];
            imageInfo[index].sampler = Attachments.GBuffer.normal.sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments.image.imageView[i];
            imageInfo[index].sampler = Attachments.image.sampler;

        index = 0;
        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = sslr.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pBufferInfo = &bufferInfo;
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = sslr.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[0];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = sslr.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[1];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = sslr.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[2];
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void SSLRGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffer)
{
    std::array<VkClearValue, 1> ClearValues{};
        ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sslr.Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sslr.PipelineLayout, 0, 1, &sslr.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
}
