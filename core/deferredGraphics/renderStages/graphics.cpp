#include "graphics.h"
#include "operations.h"
#include "object.h"
#include "deferredAttachments.h"

namespace moon::deferredGraphics {

Graphics::Graphics(
    GraphicsParameters& parameters,
    const interfaces::Objects* object,
    const interfaces::Lights* lightSources,
    const interfaces::DepthMaps* depthMaps)
    :   Workflow(parameters.imageInfo, parameters.shadersPath), parameters(parameters),
        base(parameters.imageInfo, parameters, object),
        outlining(base),
        lighting(parameters.imageInfo, parameters, lightSources, depthMaps),
        ambientLighting(lighting)
{}

void Graphics::createAttachments(moon::utils::AttachmentsDatabase& aDatabase)
{
    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    deferredAttachments.image = utils::Attachments(physicalDevice, device, parameters.imageInfo, usage);
    deferredAttachments.blur = utils::Attachments(physicalDevice, device, parameters.imageInfo, usage);
    deferredAttachments.bloom = utils::Attachments(physicalDevice, device, parameters.imageInfo, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    moon::utils::ImageInfo f32Image = { parameters.imageInfo.Count, VK_FORMAT_R32G32B32A32_SFLOAT, parameters.imageInfo.Extent, parameters.imageInfo.Samples };
    deferredAttachments.GBuffer.position = utils::Attachments(physicalDevice, device, f32Image, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
    deferredAttachments.GBuffer.normal = utils::Attachments(physicalDevice, device, f32Image, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);

    moon::utils::ImageInfo u8Image = { parameters.imageInfo.Count, VK_FORMAT_R8G8B8A8_UNORM, parameters.imageInfo.Extent, parameters.imageInfo.Samples };
    deferredAttachments.GBuffer.color = utils::Attachments(physicalDevice, device, u8Image, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, {{0.0f,0.0f,0.0f,1.0f}});

    moon::utils::ImageInfo depthImage = { parameters.imageInfo.Count, moon::utils::image::depthStencilFormat(physicalDevice), parameters.imageInfo.Extent, parameters.imageInfo.Samples };
    deferredAttachments.GBuffer.depth = utils::Attachments(physicalDevice, device, depthImage, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, { { 1.0f, 0 } });

    aDatabase.addAttachmentData((!parameters.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.parameters.transparencyNumber) + ".") + parameters.out.image, parameters.enable, &deferredAttachments.image);
    aDatabase.addAttachmentData((!parameters.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.parameters.transparencyNumber) + ".") + parameters.out.blur, parameters.enable, &deferredAttachments.blur);
    aDatabase.addAttachmentData((!parameters.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.parameters.transparencyNumber) + ".") + parameters.out.bloom, parameters.enable, &deferredAttachments.bloom);
    aDatabase.addAttachmentData((!parameters.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.parameters.transparencyNumber) + ".") + parameters.out.position, parameters.enable, &deferredAttachments.GBuffer.position);
    aDatabase.addAttachmentData((!parameters.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.parameters.transparencyNumber) + ".") + parameters.out.normal, parameters.enable, &deferredAttachments.GBuffer.normal);
    aDatabase.addAttachmentData((!parameters.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.parameters.transparencyNumber) + ".") + parameters.out.color, parameters.enable, &deferredAttachments.GBuffer.color);
    aDatabase.addAttachmentData((!parameters.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.parameters.transparencyNumber) + ".") + parameters.out.depth, parameters.enable, &deferredAttachments.GBuffer.depth);
}

void Graphics::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(deferredAttachments.image.format()),
        moon::utils::Attachments::imageDescription(deferredAttachments.blur.format()),
        moon::utils::Attachments::imageDescription(deferredAttachments.bloom.format()),
        moon::utils::Attachments::imageDescription(deferredAttachments.GBuffer.position.format()),
        moon::utils::Attachments::imageDescription(deferredAttachments.GBuffer.normal.format()),
        moon::utils::Attachments::imageDescription(deferredAttachments.GBuffer.color.format()),
        moon::utils::Attachments::depthStencilDescription(deferredAttachments.GBuffer.depth.format())
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

    utils::vkDefault::RenderPass::SubpassDescriptions subpasses;
    auto refIt = attachmentRef.begin(), inRefIt = inAttachmentRef.begin(), depthIt = depthAttachmentRef.begin();
    for(;refIt != attachmentRef.end() && inRefIt != inAttachmentRef.end() && depthIt != depthAttachmentRef.end(); refIt++, inRefIt++, depthIt++){
        subpasses.push_back(VkSubpassDescription{});
        subpasses.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpasses.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
        subpasses.back().pColorAttachments = refIt->data();
        subpasses.back().inputAttachmentCount = static_cast<uint32_t>(inRefIt->size());
        subpasses.back().pInputAttachments = inRefIt->data();
        subpasses.back().pDepthStencilAttachment = depthIt->data();
    }

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies.back().dstSubpass = 0;
        dependencies.back().srcStageMask =    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT|
                                            VK_PIPELINE_STAGE_HOST_BIT;
        dependencies.back().srcAccessMask =   VK_ACCESS_HOST_READ_BIT;
        dependencies.back().dstStageMask =    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|
                                            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|
                                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT|
                                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask =   VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT|
                                            VK_ACCESS_UNIFORM_READ_BIT|
                                            VK_ACCESS_INDEX_READ_BIT|
                                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = 0;
        dependencies.back().dstSubpass = 1;
        dependencies.back().srcStageMask =  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies.back().dstStageMask =  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|
                                            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|
                                            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
                                            VK_ACCESS_INPUT_ATTACHMENT_READ_BIT|
                                            VK_ACCESS_UNIFORM_READ_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpasses, dependencies);
}

void Graphics::createFramebuffers()
{
    framebuffers.resize(parameters.imageInfo.Count);
    for (size_t imageIndex = 0; imageIndex < parameters.imageInfo.Count; imageIndex++){
        std::vector<VkImageView> attachments;
        for(uint32_t attIndex = 0; attIndex < static_cast<uint32_t>(deferredAttachments.size()); attIndex++){
            attachments.push_back(deferredAttachments[attIndex].imageView(imageIndex));
        }

        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = parameters.imageInfo.Extent.width;
            framebufferInfo.height = parameters.imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[imageIndex] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void Graphics::createPipelines() {
    base.create(parameters.shadersPath, device, renderPass);
    outlining.create(parameters.shadersPath, device, renderPass);
    lighting.create(parameters.shadersPath, device, renderPass);
    ambientLighting.create(parameters.shadersPath, device, renderPass);
}

void Graphics::create(moon::utils::AttachmentsDatabase& aDatabase)
{
    if(parameters.enable && !created){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        created = true;
    }
}

void Graphics::updateDescriptorSets(
    const moon::utils::BuffersDatabase& bDatabase,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if (!parameters.enable || !created) return;

    base.updateDescriptorSets(device, bDatabase, aDatabase);
    lighting.updateDescriptorSets(device, bDatabase, aDatabase);
}

void Graphics::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    const std::vector<VkClearValue> clearValues = deferredAttachments.clearValues();
    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        uint32_t primitiveCount = 0;

        base.render(frameNumber,commandBuffers[frameNumber], primitiveCount);
        outlining.render(frameNumber,commandBuffers[frameNumber]);

    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);

        lighting.render(frameNumber,commandBuffers[frameNumber]);
        ambientLighting.render(frameNumber,commandBuffers[frameNumber]);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

}
