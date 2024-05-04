#include "graphics.h"
#include "operations.h"
#include "object.h"
#include "deferredAttachments.h"

graphics::graphics(
    graphicsParameters parameters,
    bool enable,
    bool enableTransparency,
    bool transparencyPass,
    uint32_t transparencyNumber,
    std::vector<moon::interfaces::Object*>* object, std::vector<moon::interfaces::Light*>* lightSources,
    std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps) :
    parameters(parameters), enable(enable)
{
    base.enableTransparency = enableTransparency;
    base.transparencyPass = transparencyPass;
    base.transparencyNumber = transparencyNumber;
    base.objects = object;
    lighting.lightSources = lightSources;
    lighting.depthMaps = depthMaps;

    outlining.Parent = &base;
    ambientLighting.Parent = &lighting;
}

graphics& graphics::setMinAmbientFactor(const float& minAmbientFactor){
    ambientLighting.minAmbientFactor = minAmbientFactor;
    return *this;
}

void graphics::destroy()
{
    base.Destroy(device);
    outlining.DestroyPipeline(device);
    lighting.Destroy(device);
    ambientLighting.DestroyPipeline(device);

    moon::workflows::Workflow::destroy();

    deferredAttachments.deleteAttachment(device);
    deferredAttachments.deleteSampler(device);
}


void graphics::createAttachments(moon::utils::AttachmentsDatabase& aDatabase)
{
    auto createAttachments = [](VkPhysicalDevice physicalDevice, VkDevice device, const moon::utils::ImageInfo& image, DeferredAttachments* pAttachments){
        VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        pAttachments->image.create(physicalDevice, device, image.Format, usage, image.Extent, image.Count);
        pAttachments->blur.create(physicalDevice, device, image.Format, usage, image.Extent, image.Count);
        pAttachments->bloom.create(physicalDevice, device, image.Format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, image.Extent, image.Count);
        pAttachments->GBuffer.position.create(physicalDevice, device, VK_FORMAT_R32G32B32A32_SFLOAT, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, image.Extent, image.Count);
        pAttachments->GBuffer.normal.create(physicalDevice, device, VK_FORMAT_R32G32B32A32_SFLOAT, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, image.Extent, image.Count);
        pAttachments->GBuffer.color.create(physicalDevice, device, VK_FORMAT_R8G8B8A8_UNORM, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, image.Extent, image.Count);
        pAttachments->GBuffer.depth.createDepth(physicalDevice, device, moon::utils::image::depthStencilFormat(physicalDevice), VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, image.Extent, image.Count);

        VkSamplerCreateInfo SamplerInfo{};
        SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        CHECK(vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->image.sampler));
        CHECK(vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->blur.sampler));
        CHECK(vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->bloom.sampler));
        CHECK(vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->GBuffer.position.sampler));
        CHECK(vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->GBuffer.normal.sampler));
        CHECK(vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->GBuffer.color.sampler));
        CHECK(vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->GBuffer.depth.sampler));
    };

    createAttachments(physicalDevice, device, image, &deferredAttachments);

    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.image, enable, &deferredAttachments.image);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.blur, enable, &deferredAttachments.blur);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.bloom, enable, &deferredAttachments.bloom);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.position, enable, &deferredAttachments.GBuffer.position);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.normal, enable, &deferredAttachments.GBuffer.normal);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.color, enable, &deferredAttachments.GBuffer.color);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.depth, enable, &deferredAttachments.GBuffer.depth);
}

void graphics::createRenderPass()
{
    std::vector<VkAttachmentDescription> attachments = {
        moon::utils::Attachments::imageDescription(deferredAttachments.image.format),
        moon::utils::Attachments::imageDescription(deferredAttachments.blur.format),
        moon::utils::Attachments::imageDescription(deferredAttachments.bloom.format),
        moon::utils::Attachments::imageDescription(deferredAttachments.GBuffer.position.format),
        moon::utils::Attachments::imageDescription(deferredAttachments.GBuffer.normal.format),
        moon::utils::Attachments::imageDescription(deferredAttachments.GBuffer.color.format),
        moon::utils::Attachments::depthStencilDescription(deferredAttachments.GBuffer.depth.format)
    };

    uint32_t gOffset = DeferredAttachments::GBufferOffset();

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::positionIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::normalIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::colorIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{DeferredAttachments::imageIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{DeferredAttachments::blurIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{DeferredAttachments::bloomIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    std::vector<std::vector<VkAttachmentReference>> inAttachmentRef;
    inAttachmentRef.push_back(std::vector<VkAttachmentReference>());
    inAttachmentRef.push_back(std::vector<VkAttachmentReference>());
        inAttachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::positionIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        inAttachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::normalIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        inAttachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::colorIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        inAttachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::depthIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});

    std::vector<std::vector<VkAttachmentReference>> depthAttachmentRef;
    depthAttachmentRef.push_back(std::vector<VkAttachmentReference>());
        depthAttachmentRef.back().push_back(VkAttachmentReference{gOffset + GBufferAttachments::depthIndex(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});
    depthAttachmentRef.push_back(std::vector<VkAttachmentReference>());

    auto refIt = attachmentRef.begin(), inRefIt = inAttachmentRef.begin(), depthIt = depthAttachmentRef.begin();
    std::vector<VkSubpassDescription> subpass;
    for(;refIt != attachmentRef.end() && inRefIt != inAttachmentRef.end() && depthIt != depthAttachmentRef.end(); refIt++, inRefIt++, depthIt++){
        subpass.push_back(VkSubpassDescription{});
            subpass.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
            subpass.back().pColorAttachments = refIt->data();
            subpass.back().inputAttachmentCount = static_cast<uint32_t>(inRefIt->size());
            subpass.back().pInputAttachments = inRefIt->data();
            subpass.back().pDepthStencilAttachment = depthIt->data();
    }

    std::vector<VkSubpassDependency> dependency;
    dependency.push_back(VkSubpassDependency{});
        dependency.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.back().dstSubpass = 0;
        dependency.back().srcStageMask =    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT|
                                            VK_PIPELINE_STAGE_HOST_BIT;
        dependency.back().srcAccessMask =   VK_ACCESS_HOST_READ_BIT;
        dependency.back().dstStageMask =    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|
                                            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|
                                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT|
                                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.back().dstAccessMask =   VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT|
                                            VK_ACCESS_UNIFORM_READ_BIT|
                                            VK_ACCESS_INDEX_READ_BIT|
                                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.push_back(VkSubpassDependency{});
        dependency.back().srcSubpass = 0;
        dependency.back().dstSubpass = 1;
        dependency.back().srcStageMask =    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.back().srcAccessMask =   VK_ACCESS_MEMORY_READ_BIT;
        dependency.back().dstStageMask =    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|
                                            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|
                                            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependency.back().dstAccessMask =   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
                                            VK_ACCESS_INPUT_ATTACHMENT_READ_BIT|
                                            VK_ACCESS_UNIFORM_READ_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pDependencies = dependency.data();
    CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

void graphics::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for (size_t imageIndex = 0; imageIndex < image.Count; imageIndex++){
        std::vector<VkImageView> attachments;
        for(size_t attIndex = 0; attIndex < deferredAttachments.size(); attIndex++){
            attachments.push_back(deferredAttachments[static_cast<uint32_t>(attIndex)].instances[imageIndex].imageView);
        }

        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[imageIndex]));
    }
}

void graphics::createPipelines()
{
    base.ShadersPath = shadersPath;
    outlining.ShadersPath = shadersPath;
    lighting.ShadersPath = shadersPath;
    ambientLighting.ShadersPath = shadersPath;

    base.createDescriptorSetLayout(device);
    base.createPipeline(device,&image,renderPass);
    outlining.createPipeline(device,&image,renderPass);
    lighting.createDescriptorSetLayout(device);
    lighting.createPipeline(device,&image,renderPass);
    ambientLighting.createPipeline(device,&image,renderPass);
}

void graphics::createDescriptorPool()
{
    createBaseDescriptorPool();
    createLightingDescriptorPool();
}

void graphics::createDescriptorSets()
{
    createBaseDescriptorSets();
    createLightingDescriptorSets();
}

void graphics::create(moon::utils::AttachmentsDatabase& aDatabase)
{
    if(enable){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        createDescriptorPool();
        createDescriptorSets();
    }
}

void graphics::updateDescriptorSets(
    const moon::utils::BuffersDatabase& bDatabase,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if(!enable) return;

    updateBaseDescriptorSets(bDatabase, aDatabase);
    updateLightingDescriptorSets(bDatabase);
}

void graphics::updateCommandBuffer(uint32_t frameNumber){
    if(!enable) return;

    std::vector<VkClearValue> clearValues;
    for(size_t attIndex = 0; attIndex < deferredAttachments.size(); attIndex++){
        clearValues.push_back(deferredAttachments[static_cast<uint32_t>(attIndex)].clearValue);
    }

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        primitiveCount = 0;

        base.render(frameNumber,commandBuffers[frameNumber], primitiveCount);
        outlining.render(frameNumber,commandBuffers[frameNumber]);

    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);

        lighting.render(frameNumber,commandBuffers[frameNumber]);
        ambientLighting.render(frameNumber,commandBuffers[frameNumber]);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}
