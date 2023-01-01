#include "graphics.h"
#include "core/transformational/object.h"
#include "core/transformational/camera.h"
#include "core/texture.h"
#include "core/operations.h"
#include "../bufferObjects.h"

#include <array>
#include <vector>

deferredGraphics::deferredGraphics(){
    pAttachments.resize(7);
    outlining.Parent = &base;
    ambientLighting.Parent = &lighting;
}

void deferredGraphics::setExternalPath(const std::string &path)
{
    base.ExternalPath = path;
    outlining.ExternalPath = path;
    skybox.ExternalPath = path;
    lighting.ExternalPath = path;
    ambientLighting.ExternalPath = path;
}

void                            deferredGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
}
void                            deferredGraphics::setEmptyTexture(std::string ZERO_TEXTURE){
    this->emptyTexture = new texture(ZERO_TEXTURE);
    emptyTexture->createTextureImage(physicalDevice,device,graphicsQueue,commandPool);
    emptyTexture->createTextureImageView(device);
    emptyTexture->createTextureSampler(device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
}
void                            deferredGraphics::setCameraObject(camera* cameraObject)                 { this->cameraObject = cameraObject;}
void                            deferredGraphics::setImageProp(imageInfo* pInfo)                        { this->image = *pInfo;}

void                            deferredGraphics::setMinAmbientFactor(const float& minAmbientFactor)    { ambientLighting.minAmbientFactor = minAmbientFactor;}
void                            deferredGraphics::setScattering(const bool &enableScattering)           { lighting.enableScattering = enableScattering;}
void                            deferredGraphics::setTransparencyPass(const bool& transparencyPass)     { this->transparencyPass = transparencyPass;}

texture*                        deferredGraphics::getEmptyTexture()                                     { return emptyTexture;}
VkBuffer*                       deferredGraphics::getSceneBuffer()                                      { return base.sceneUniformBuffers.data();}

void deferredGraphics::destroyEmptyTexture(){
    if(emptyTexture){
        emptyTexture->destroy(device);
        delete emptyTexture;
    }
}

void deferredGraphics::destroy()
{
    base.Destroy(device);
    outlining.DestroyOutliningPipeline(device);
    skybox.Destroy(device);
    lighting.Destroy(device);
    ambientLighting.DestroyPipeline(device);

    if(renderPass) vkDestroyRenderPass(*device, renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        if(framebuffers[i]) vkDestroyFramebuffer(*device, framebuffers[i],nullptr);

    for(size_t i=0;i<colorAttachments.size();i++)
        colorAttachments[i].deleteAttachment(device);
}

void                            deferredGraphics::setAttachments(DeferredAttachments* Attachments)
{
    this->pAttachments[0]  = &Attachments->image           ;
    this->pAttachments[1]  = &Attachments->blur            ;
    this->pAttachments[2]  = &Attachments->bloom           ;
    this->pAttachments[3]  = &Attachments->GBuffer.position;
    this->pAttachments[4]  = &Attachments->GBuffer.normal  ;
    this->pAttachments[5]  = &Attachments->GBuffer.color   ;
    this->pAttachments[6]  = &Attachments->GBuffer.emission;
    this->depthAttachment = &Attachments->depth           ;
}

void deferredGraphics::createBufferAttachments()
{
    if(image.Samples!=VK_SAMPLE_COUNT_1_BIT)
    {
        colorAttachments.resize(7);
        for(size_t i=0;i<3;i++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                image.Samples,
                                image.Format,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                colorAttachments[i].image,
                                colorAttachments[i].imageMemory);

            createImageView(    device,
                                colorAttachments[i].image,
                                image.Format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &colorAttachments[i].imageView);
        }
        for(size_t i=3;i<5;i++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                image.Samples,
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                colorAttachments[i].image,
                                colorAttachments[i].imageMemory);

            createImageView(    device,
                                colorAttachments[i].image,
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &colorAttachments[i].imageView);
        }
        for(size_t i=5;i<colorAttachments.size();i++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                image.Samples,
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                colorAttachments[i].image,
                                colorAttachments[i].imageMemory);

            createImageView(    device,
                                colorAttachments[i].image,
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &colorAttachments[i].imageView);
        }
    }
}

void deferredGraphics::createAttachments(DeferredAttachments* pAttachments)
{
    std::vector<attachments *> attachments(7);
    attachments[0]  = &pAttachments->image           ;
    attachments[1]  = &pAttachments->blur            ;
    attachments[2]  = &pAttachments->bloom           ;
    attachments[3]  = &pAttachments->GBuffer.position;
    attachments[4]  = &pAttachments->GBuffer.normal  ;
    attachments[5]  = &pAttachments->GBuffer.color   ;
    attachments[6]  = &pAttachments->GBuffer.emission;
    attachment* depthAttachment = &pAttachments->depth;

    for(size_t i=0;i<2;i++)
    {
        attachments[i]->resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                image.Format,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                attachments[i]->image[Image],
                                attachments[i]->imageMemory[Image]);

            createImageView(    device,
                                attachments[i]->image[Image],
                                image.Format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &attachments[i]->imageView[Image]);
        }
    }
    attachments[2]->resize(image.Count);
    for(size_t Image=0; Image<image.Count; Image++)
    {
        createImage(        physicalDevice,
                            device,
                            image.Extent.width,
                            image.Extent.height,
                            1,
                            VK_SAMPLE_COUNT_1_BIT,
                            image.Format,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            attachments[2]->image[Image],
                            attachments[2]->imageMemory[Image]);

        createImageView(    device,
                            attachments[2]->image[Image],
                            image.Format,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &attachments[2]->imageView[Image]);
    }
    for(size_t i=3;i<5;i++)
    {
        attachments[i]->resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                attachments[i]->image[Image],
                                attachments[i]->imageMemory[Image]);

            createImageView(    device,
                                attachments[i]->image[Image],
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &attachments[i]->imageView[Image]);
        }
    }
    for(size_t i=5;i<attachments.size();i++)
    {
        attachments[i]->resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                attachments[i]->image[Image],
                                attachments[i]->imageMemory[Image]);

            createImageView(    device,
                                attachments[i]->image[Image],
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &attachments[i]->imageView[Image]);
        }
    }

    for(size_t i=0;i<7;i++)
    {
        VkSamplerCreateInfo SamplerInfo{};
            SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        vkCreateSampler(*device, &SamplerInfo, nullptr, &attachments[i]->sampler);
    }

    createImage(        physicalDevice,
                        device,
                        image.Extent.width,
                        image.Extent.height,
                        1,
                        image.Samples,
                        findDepthStencilFormat(physicalDevice),
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        depthAttachment->image,
                        depthAttachment->imageMemory);

    createImageView(    device,
                        depthAttachment->image,
                        findDepthStencilFormat(physicalDevice),
                        VK_IMAGE_ASPECT_DEPTH_BIT,
                        1,
                        &depthAttachment->imageView);

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
    vkCreateSampler(*device, &SamplerInfo, nullptr, &depthAttachment->sampler);
}

void deferredGraphics::createRenderPass()
{
    if(image.Samples==VK_SAMPLE_COUNT_1_BIT) oneSampleRenderPass();
    else                                     multiSampleRenderPass();
}
    void deferredGraphics::oneSampleRenderPass()
    {
        std::vector<VkAttachmentDescription> attachments;
        VkAttachmentDescription colorAttachment{};
            colorAttachment.format = image.Format;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(colorAttachment);
            colorAttachment.format = image.Format;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(colorAttachment);
            colorAttachment.format = image.Format;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(colorAttachment);
        for(size_t i=3;i<5;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        for(size_t i=5;i<pAttachments.size();i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthStencilFormat(physicalDevice);
            depthAttachment.samples = image.Samples;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(depthAttachment);

        uint32_t index = 3;
        std::vector<VkAttachmentReference> firstAttachmentRef(4);
            for (size_t i=0;i<firstAttachmentRef.size();i++)
            {
                firstAttachmentRef[i].attachment = index;
                firstAttachmentRef[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                index++;
            }
        VkAttachmentReference firstDepthAttachmentRef{};
            firstDepthAttachmentRef.attachment = index++;
            firstDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        index = 0;
        std::vector<VkAttachmentReference> secondAttachmentRef(3);
            secondAttachmentRef[index].attachment = 0;
            secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef[index].attachment = 1;
            secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef[index].attachment = 2;
            secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        index = 0;
        std::vector<VkAttachmentReference> secondInAttachmentRef(5);
            secondInAttachmentRef[index].attachment = 3;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef[index].attachment = 4;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef[index].attachment = 5;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef[index].attachment = 6;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef[index].attachment = 7;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        index = 0;
        std::vector<VkSubpassDescription> subpass(2);
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
        std::vector<VkSubpassDependency> dependency(2);
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
    void deferredGraphics::multiSampleRenderPass()
    {
        std::vector<VkAttachmentDescription> attachments;
        for(size_t i=0;i<3;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = image.Format;
                colorAttachment.samples = image.Samples;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        for(size_t i=3;i<5;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT;
                colorAttachment.samples = image.Samples;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        for(size_t i=5;i<colorAttachments.size();i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
                colorAttachment.samples = image.Samples;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments.push_back(colorAttachment);
        }

        VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthStencilFormat(physicalDevice);
            depthAttachment.samples = image.Samples;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAttachment);

        for(size_t i=0;i<2;i++)
        {
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = image.Format;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachmentResolve);
        }
        VkAttachmentDescription colorAttachmentResolve{};
            colorAttachmentResolve.format = image.Format;
            colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attachments.push_back(colorAttachmentResolve);
        for(size_t i=3;i<5;i++)
        {
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = VK_FORMAT_R32G32B32A32_SFLOAT;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachmentResolve);
        }
        for(size_t i=5;i<pAttachments.size();i++)
        {
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = VK_FORMAT_R8G8B8A8_UNORM;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachmentResolve);
        }

        uint32_t index = 3;
        std::vector<VkAttachmentReference> firstAttachmentRef(4);
            for (size_t i=0;i<firstAttachmentRef.size();i++)
            {
                firstAttachmentRef[i].attachment = index;
                firstAttachmentRef[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                index++;
            }

        VkAttachmentReference firstDepthAttachmentRef{};
            firstDepthAttachmentRef.attachment = index++;
            firstDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        index = colorAttachments.size()+4;
        std::vector<VkAttachmentReference> firstResolveRef(4);
            for (size_t i=0;i<firstResolveRef.size();i++)
            {
                firstResolveRef[i].attachment = index;
                firstResolveRef[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                index++;
            }

        index = 0;
        std::vector<VkAttachmentReference> secondAttachmentRef(3);
            secondAttachmentRef[index].attachment = 0;
            secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef[index].attachment = 1;
            secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef[index].attachment = 2;
            secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        index = 0;
        std::vector<VkAttachmentReference> secondResolveRef(3);
            secondResolveRef[index].attachment = colorAttachments.size()+1;
            secondResolveRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondResolveRef[index].attachment = colorAttachments.size()+2;
            secondResolveRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondResolveRef[index].attachment = colorAttachments.size()+3;
            secondResolveRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        index = 0;
        std::vector<VkAttachmentReference> secondInAttachmentRef(4);
            secondInAttachmentRef[index].attachment = colorAttachments.size()+4;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef[index].attachment = colorAttachments.size()+5;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef[index].attachment = colorAttachments.size()+6;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef[index].attachment = colorAttachments.size()+7;
            secondInAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        index = 0;
        std::vector<VkSubpassDescription> subpass(2);
            subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass[index].colorAttachmentCount = static_cast<uint32_t>(firstAttachmentRef.size());
            subpass[index].pColorAttachments = firstAttachmentRef.data();
            subpass[index].pDepthStencilAttachment = &firstDepthAttachmentRef;
            subpass[index].pResolveAttachments = firstResolveRef.data();
        index++;
            subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass[index].colorAttachmentCount = static_cast<uint32_t>(secondAttachmentRef.size());
            subpass[index].pColorAttachments = secondAttachmentRef.data();
            subpass[index].inputAttachmentCount = static_cast<uint32_t>(secondInAttachmentRef.size());
            subpass[index].pInputAttachments = secondInAttachmentRef.data();
            subpass[index].pDepthStencilAttachment = &firstDepthAttachmentRef;
            subpass[index].pResolveAttachments = secondResolveRef.data();

        index = 0;
        std::vector<VkSubpassDependency> dependency(2);
            dependency[index].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency[index].dstSubpass = 0;
            dependency[index].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependency[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        index++;
            dependency[index].srcSubpass = 0;
            dependency[index].dstSubpass = 1;
            dependency[index].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency[index].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency[index].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[index].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

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
    if(image.Samples == VK_SAMPLE_COUNT_1_BIT){
        oneSampleFrameBuffer();
    }else{
        multiSampleFrameBuffer();
    }
}
    void deferredGraphics::oneSampleFrameBuffer()
    {
        framebuffers.resize(image.Count);
        for (size_t Image = 0; Image < image.Count; Image++)
        {
            std::vector<VkImageView> attachments;
            for(size_t i=0;i<pAttachments.size();i++)
                attachments.push_back(pAttachments[i]->imageView[Image]);
            attachments.push_back(depthAttachment->imageView);

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
    void deferredGraphics::multiSampleFrameBuffer()
    {
        framebuffers.resize(image.Count);
        for (size_t Image = 0; Image < image.Count; Image++)
        {
            std::vector<VkImageView> attachments;
            for(size_t i=0;i<colorAttachments.size();i++)
                attachments.push_back(colorAttachments[i].imageView);
            attachments.push_back(depthAttachment->imageView);
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
    base.createUniformBuffers(physicalDevice,device,image.Count);
    outlining.createOutliningPipeline(device,&image,&renderPass);
    skybox.createDescriptorSetLayout(device);
    skybox.createPipeline(device,&image,&renderPass);
    skybox.createUniformBuffers(physicalDevice,device,image.Count);
    lighting.createDescriptorSetLayout(device);
    lighting.createPipeline(device,&image,&renderPass);
    lighting.createUniformBuffers(physicalDevice,device,image.Count);
    ambientLighting.createPipeline(device,&image,&renderPass);
}

void deferredGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    std::vector<VkClearValue> clearValues;
    if(image.Samples == VK_SAMPLE_COUNT_1_BIT){
        clearValues.resize(8);
        for(size_t i=0;i<7;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[7].depthStencil = {1.0f, 0};
    }else{
        clearValues.resize(15);
        for(size_t i=0;i<7;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[7].depthStencil = {1.0f, 0};
        for(size_t i=8;i<15;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }

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

        skybox.render(frameNumber,commandBuffers);
        base.render(frameNumber,commandBuffers, primitiveCount);
        outlining.render(frameNumber,commandBuffers);

    vkCmdNextSubpass(commandBuffers, VK_SUBPASS_CONTENTS_INLINE);

        lighting.render(frameNumber,commandBuffers);
        ambientLighting.render(frameNumber,commandBuffers);

    vkCmdEndRenderPass(commandBuffers);
}

void deferredGraphics::updateUniformBuffer(uint32_t currentImage)
{
    void* data;

    UniformBufferObject baseUBO{};
        baseUBO.view = cameraObject->getViewMatrix();
        baseUBO.proj = cameraObject->getProjMatrix();
        baseUBO.eyePosition = glm::vec4(cameraObject->getTranslation(), 1.0);
        baseUBO.enableTransparency = transparencyPass ? 1.0 : 0.0;
    vkMapMemory(*device, base.sceneUniformBuffersMemory[currentImage], 0, sizeof(baseUBO), 0, &data);
        memcpy(data, &baseUBO, sizeof(baseUBO));
    vkUnmapMemory(*device, base.sceneUniformBuffersMemory[currentImage]);

    vkMapMemory(*device, lighting.uniformBuffersMemory[currentImage], 0, sizeof(baseUBO), 0, &data);
        memcpy(data, &baseUBO, sizeof(baseUBO));
    vkUnmapMemory(*device, lighting.uniformBuffersMemory[currentImage]);
}

void deferredGraphics::updateSkyboxUniformBuffer(uint32_t currentImage)
{
    if(skybox.objects.size()!=0)
    {
        void* data;

        SkyboxUniformBufferObject skyboxUBO{};
            skyboxUBO.view = cameraObject->getViewMatrix();
            skyboxUBO.proj = cameraObject->getProjMatrix();
            skyboxUBO.model = glm::translate(glm::mat4x4(1.0f),cameraObject->getTranslation())*skybox.objects[0]->ModelMatrix();
        vkMapMemory(*device, this->skybox.uniformBuffersMemory[currentImage], 0, sizeof(skyboxUBO), 0, &data);
            memcpy(data, &skyboxUBO, sizeof(skyboxUBO));
        vkUnmapMemory(*device, this->skybox.uniformBuffersMemory[currentImage]);
    }
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

void deferredGraphics::bindSkyBoxObject(object *newObject, const std::vector<std::string>& TEXTURE_PATH)
{
    skybox.texture = new cubeTexture(TEXTURE_PATH);
    skybox.texture->setMipLevel(0.0f);
    skybox.texture->createTextureImage(physicalDevice,device,graphicsQueue,commandPool);
    skybox.texture->createTextureImageView(device);
    skybox.texture->createTextureSampler(device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    skybox.objects.push_back(newObject);
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

bool deferredGraphics::removeSkyBoxObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<skybox.objects.size(); index++){
        if(object==skybox.objects[index]){
            skybox.texture->destroy(device);
            delete skybox.texture;
            skybox.objects.erase(skybox.objects.begin()+index);
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
