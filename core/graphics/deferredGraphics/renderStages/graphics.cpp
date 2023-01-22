#include "graphics.h"
#include "core/transformational/object.h"
#include "core/transformational/camera.h"
#include "core/texture.h"
#include "core/operations.h"

#include <array>
#include <vector>

deferredGraphics::deferredGraphics(){
    outlining.Parent = &base;
    ambientLighting.Parent = &lighting;
}

void deferredGraphics::setExternalPath(const std::string &path){
    base.ExternalPath = path;
    outlining.ExternalPath = path;
    lighting.ExternalPath = path;
    ambientLighting.ExternalPath = path;
}

void deferredGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool){
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
}

void deferredGraphics::setEmptyTexture(texture* emptyTexture){
    this->emptyTexture = emptyTexture;
}

void deferredGraphics::setImageProp(imageInfo* pInfo)                        { this->image = *pInfo;}
void deferredGraphics::setMinAmbientFactor(const float& minAmbientFactor)    { ambientLighting.minAmbientFactor = minAmbientFactor;}
void deferredGraphics::setScattering(const bool &enableScattering)           { lighting.enableScattering = enableScattering;}
void deferredGraphics::setTransparencyPass(const bool& transparencyPass)     { base.transparencyPass = transparencyPass;}

void deferredGraphics::destroy()
{
    base.Destroy(device);
    outlining.DestroyOutliningPipeline(device);
    lighting.Destroy(device);
    ambientLighting.DestroyPipeline(device);

    if(renderPass) {vkDestroyRenderPass(*device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(size_t i = 0; i< framebuffers.size();i++)
        if(framebuffers[i]) vkDestroyFramebuffer(*device, framebuffers[i],nullptr);
    framebuffers.resize(0);
}

void deferredGraphics::setAttachments(DeferredAttachments* Attachments)
{
    pAttachments.resize(8);
    this->pAttachments[0]  = &Attachments->image           ;
    this->pAttachments[1]  = &Attachments->blur            ;
    this->pAttachments[2]  = &Attachments->bloom           ;
    this->pAttachments[3]  = &Attachments->depth           ;
    this->pAttachments[4]  = &Attachments->GBuffer.position;
    this->pAttachments[5]  = &Attachments->GBuffer.normal  ;
    this->pAttachments[6]  = &Attachments->GBuffer.color   ;
    this->pAttachments[7]  = &Attachments->GBuffer.emission;
}

void deferredGraphics::createAttachments(DeferredAttachments* pAttachments)
{
    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    pAttachments->image.create(physicalDevice, device, image.Format, usage, image.Extent, image.Count);
    pAttachments->blur.create(physicalDevice, device, image.Format, usage, image.Extent, image.Count);
    pAttachments->bloom.create(physicalDevice, device, image.Format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, image.Extent, image.Count);
    pAttachments->depth.createDepth(physicalDevice, device, findDepthStencilFormat(physicalDevice), VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, image.Extent, image.Count);
    pAttachments->GBuffer.position.create(physicalDevice, device, VK_FORMAT_R32G32B32A32_SFLOAT, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, image.Extent, image.Count);
    pAttachments->GBuffer.normal.create(physicalDevice, device, VK_FORMAT_R32G32B32A32_SFLOAT, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, image.Extent, image.Count);
    pAttachments->GBuffer.color.create(physicalDevice, device, VK_FORMAT_R8G8B8A8_UNORM, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, image.Extent, image.Count);
    pAttachments->GBuffer.emission.create(physicalDevice, device, VK_FORMAT_R8G8B8A8_UNORM, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, image.Extent, image.Count);

    VkSamplerCreateInfo SamplerInfo{};
        SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->image.sampler);
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->blur.sampler);
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->bloom.sampler);
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->depth.sampler);
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->GBuffer.position.sampler);
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->GBuffer.normal.sampler);
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->GBuffer.color.sampler);
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->GBuffer.emission.sampler);
}

void deferredGraphics::createRenderPass()
{
    std::vector<VkAttachmentDescription> attachments(pAttachments.size());
    attachments[0] = attachments::imageDescription(pAttachments[0]->format);
    attachments[1] = attachments::imageDescription(pAttachments[1]->format);
    attachments[2] = attachments::imageDescription(pAttachments[2]->format);
    attachments[3] = attachments::depthStencilDescription(pAttachments[3]->format);
    attachments[4] = attachments::imageDescription(pAttachments[4]->format);
    attachments[5] = attachments::imageDescription(pAttachments[5]->format);
    attachments[6] = attachments::imageDescription(pAttachments[6]->format);
    attachments[7] = attachments::imageDescription(pAttachments[7]->format);

    VkAttachmentReference firstDepthAttachmentRef{};
        firstDepthAttachmentRef.attachment = DeferredAttachments::getGBufferOffset() - 1;
        firstDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::vector<VkAttachmentReference> firstAttachmentRef(pAttachments.size() - DeferredAttachments::getGBufferOffset());
    for(size_t index=0; index<firstAttachmentRef.size(); index++){
        firstAttachmentRef[index].attachment = DeferredAttachments::getGBufferOffset() + index;
        firstAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    std::vector<VkAttachmentReference> secondAttachmentRef(DeferredAttachments::getGBufferOffset() - 1);
    for(size_t index=0; index<secondAttachmentRef.size(); index++){
        secondAttachmentRef[index].attachment = index;
        secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    std::vector<VkAttachmentReference> secondInAttachmentRef(pAttachments.size() - DeferredAttachments::getGBufferOffset() + 1);
    for(size_t index=0; index<secondInAttachmentRef.size(); index++){
        secondInAttachmentRef[index].attachment = DeferredAttachments::getGBufferOffset() - 1 + index;
        secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    uint32_t index = 0;
    std::array<VkSubpassDescription,2> subpass{};
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(firstAttachmentRef.size());
        subpass[index].pColorAttachments = firstAttachmentRef.data();
        subpass[index].pDepthStencilAttachment = &firstDepthAttachmentRef;
    index++;
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(secondAttachmentRef.size());
        subpass[index].pColorAttachments = secondAttachmentRef.data();
        subpass[index].inputAttachmentCount = static_cast<uint32_t>(secondInAttachmentRef.size());
        subpass[index].pInputAttachments = secondInAttachmentRef.data();
        subpass[index].pDepthStencilAttachment = nullptr;

    index = 0;
    std::array<VkSubpassDependency, 2> dependency{};
        dependency[index].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency[index].dstSubpass = 0;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    index++;
        dependency[index].srcSubpass = 0;
        dependency[index].dstSubpass = 1;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pDependencies = dependency.data();
    vkCreateRenderPass(*device, &renderPassInfo, nullptr, &renderPass);
}

void deferredGraphics::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for (size_t Image = 0; Image < image.Count; Image++)
    {
        std::vector<VkImageView> attachments;
        for(size_t i=0;i<pAttachments.size();i++)
            attachments.push_back(pAttachments[i]->imageView[Image]);

        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[Image]);
    }
}

void deferredGraphics::createPipelines()
{
    base.createDescriptorSetLayout(device);
    base.createPipeline(device,&image,&renderPass);
    outlining.createOutliningPipeline(device,&image,&renderPass);
    lighting.createDescriptorSetLayout(device);
    lighting.createPipeline(device,&image,&renderPass);
    ambientLighting.createPipeline(device,&image,&renderPass);
}

void deferredGraphics::createDescriptorPool()
{
    createBaseDescriptorPool();
    createLightingDescriptorPool();
}

void deferredGraphics::createDescriptorSets()
{
    createBaseDescriptorSets();
    createLightingDescriptorSets();
}

void deferredGraphics::updateDescriptorSets(attachments* depthAttachment, VkBuffer* storageBuffers, camera* cameraObject)
{
    updateBaseDescriptorSets(depthAttachment, storageBuffers, cameraObject);
    updateLightingDescriptorSets(cameraObject);
}

void deferredGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    std::vector<VkClearValue> clearValues(pAttachments.size());
    for(size_t i=0;i<clearValues.size();i++)
        clearValues[i] = pAttachments[i]->clearValue;

    VkRenderPassBeginInfo drawRenderPassInfo{};
        drawRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        drawRenderPassInfo.renderPass = renderPass;
        drawRenderPassInfo.framebuffer = framebuffers[frameNumber];
        drawRenderPassInfo.renderArea.offset = {0, 0};
        drawRenderPassInfo.renderArea.extent = image.Extent;
        drawRenderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        drawRenderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers, &drawRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        primitiveCount = 0;

        base.render(frameNumber,commandBuffers, primitiveCount);
        outlining.render(frameNumber,commandBuffers);

    vkCmdNextSubpass(commandBuffers, VK_SUBPASS_CONTENTS_INLINE);

        lighting.render(frameNumber,commandBuffers);
        ambientLighting.render(frameNumber,commandBuffers);

        vkCmdEndRenderPass(commandBuffers);
}

void deferredGraphics::updateObjectUniformBuffer(uint32_t currentImage)
{
    for(size_t i=0;i<base.objects.size();i++)
        base.objects[i]->updateUniformBuffer(device,currentImage);
}

void deferredGraphics::bindBaseObject(object *newObject)
{
    base.objects.push_back(newObject);
}

bool deferredGraphics::removeBaseObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<base.objects.size(); index++){
        if(object==base.objects[index]){
            base.objects.erase(base.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

void deferredGraphics::addLightSource(light* lightSource)
{
    lighting.lightSources.push_back(lightSource);
}

void deferredGraphics::removeLightSource(light* lightSource)
{
    for(uint32_t index = 0; index<lighting.lightSources.size(); index++){
        if(lightSource==lighting.lightSources[index]){
            lighting.lightSources.erase(lighting.lightSources.begin()+index);
        }
    }
}
