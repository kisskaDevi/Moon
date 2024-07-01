#include "graphics.h"
#include "operations.h"
#include "object.h"
#include "deferredAttachments.h"

namespace moon::deferredGraphics {

Graphics::Graphics(
    const moon::utils::ImageInfo& imageInfo,
    const std::filesystem::path& shadersPath,
    GraphicsParameters parameters,
    bool enable,
    bool enableTransparency,
    bool transparencyPass,
    uint32_t transparencyNumber,
    std::vector<moon::interfaces::Object*>* object, std::vector<moon::interfaces::Light*>* lightSources,
    std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps) :
    Workflow(imageInfo, shadersPath),
    parameters(parameters),
    enable(enable),
    base(transparencyPass, enableTransparency, transparencyNumber, this->imageInfo, this->parameters),
    outlining(base),
    lighting(this->imageInfo, this->parameters),
    ambientLighting(lighting)
{
    base.objects = object;

    lighting.lightSources = lightSources;
    lighting.depthMaps = depthMaps;
}

Graphics& Graphics::setMinAmbientFactor(const float& minAmbientFactor){
    ambientLighting.minAmbientFactor = minAmbientFactor;
    return *this;
}

void Graphics::createAttachments(moon::utils::AttachmentsDatabase& aDatabase)
{
    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    deferredAttachments.image.create(physicalDevice, device, imageInfo, usage);
    deferredAttachments.blur.create(physicalDevice, device, imageInfo, usage);
    deferredAttachments.bloom.create(physicalDevice, device, imageInfo, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    moon::utils::ImageInfo f32Image = { imageInfo.Count, VK_FORMAT_R32G32B32A32_SFLOAT, imageInfo.Extent, imageInfo.Samples };
    deferredAttachments.GBuffer.position.create(physicalDevice, device, f32Image, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
    deferredAttachments.GBuffer.normal.create(physicalDevice, device, f32Image, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);

    moon::utils::ImageInfo u8Image = { imageInfo.Count, VK_FORMAT_R8G8B8A8_UNORM, imageInfo.Extent, imageInfo.Samples };
    deferredAttachments.GBuffer.color.create(physicalDevice, device, u8Image, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, {{0.0f,0.0f,0.0f,1.0f}});

    moon::utils::ImageInfo depthImage = { imageInfo.Count, moon::utils::image::depthStencilFormat(physicalDevice), imageInfo.Extent, imageInfo.Samples };
    deferredAttachments.GBuffer.depth.create(physicalDevice, device, depthImage, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, { { 1.0f, 0 } });

    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.image, enable, &deferredAttachments.image);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.blur, enable, &deferredAttachments.blur);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.bloom, enable, &deferredAttachments.bloom);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.position, enable, &deferredAttachments.GBuffer.position);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.normal, enable, &deferredAttachments.GBuffer.normal);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.color, enable, &deferredAttachments.GBuffer.color);
    aDatabase.addAttachmentData((!base.transparencyPass ? "" : parameters.out.transparency + std::to_string(base.transparencyNumber) + ".") + parameters.out.depth, enable, &deferredAttachments.GBuffer.depth);
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

    CHECK(renderPass.create(device, attachments, subpasses, dependencies));
}

void Graphics::createFramebuffers()
{
    framebuffers.resize(imageInfo.Count);
    for (size_t imageIndex = 0; imageIndex < imageInfo.Count; imageIndex++){
        std::vector<VkImageView> attachments;
        for(uint32_t attIndex = 0; attIndex < static_cast<uint32_t>(deferredAttachments.size()); attIndex++){
            attachments.push_back(deferredAttachments[attIndex].imageView(imageIndex));
        }

        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = imageInfo.Extent.width;
            framebufferInfo.height = imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        CHECK(framebuffers[imageIndex].create(device, framebufferInfo));
    }
}

void Graphics::createPipelines() {
    base.create(shadersPath, device, renderPass);
    outlining.create(shadersPath, device, renderPass);
    lighting.create(shadersPath, device, renderPass);
    ambientLighting.create(shadersPath, device, renderPass);
}

void Graphics::create(moon::utils::AttachmentsDatabase& aDatabase)
{
    if(enable){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        createPipelines();
    }
}

void Graphics::updateDescriptorSets(
    const moon::utils::BuffersDatabase& bDatabase,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if(!enable) return;

    base.updateDescriptorSets(device, bDatabase, aDatabase);
    lighting.updateDescriptorSets(device, bDatabase, aDatabase);
}

void Graphics::updateCommandBuffer(uint32_t frameNumber){
    if(!enable) return;

    const std::vector<VkClearValue> clearValues = deferredAttachments.clearValues();
    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = imageInfo.Extent;
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

}
