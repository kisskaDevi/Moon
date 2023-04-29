#include "deferredGraphics.h"
#include "operations.h"
#include "texture.h"
#include "node.h"
#include "model.h"
#include "light.h"
#include "object.h"
#include "camera.h"

#include <cstring>

deferredGraphics::deferredGraphics(const std::string& ExternalPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples):
    ExternalPath(ExternalPath), extent(extent), MSAASamples(MSAASamples)
{
    DeferredGraphics.setExternalPath(ExternalPath);
    PostProcessing.setExternalPath(ExternalPath);
    Filter.setExternalPath(ExternalPath);
    SSLR.setExternalPath(ExternalPath);
    SSAO.setExternalPath(ExternalPath);
    Skybox.setExternalPath(ExternalPath);
    Shadow.setExternalPath(ExternalPath);
    LayersCombiner.setExternalPath(ExternalPath);
    Blur.setExternalPath(ExternalPath);

    TransparentLayers.resize(TransparentLayersCount);
    for(auto& layer: TransparentLayers){
        layer.setExternalPath(ExternalPath);
    }
}

deferredGraphics::~deferredGraphics()
{}

void deferredGraphics::destroyEmptyTextures()
{
    if(emptyTexture){
        emptyTexture->destroy(device.getLogical());
        emptyTexture = nullptr;
    }
}

void deferredGraphics::freeCommandBuffers()
{
    Blur.freeCommandBuffer(commandPool);
    Filter.freeCommandBuffer(commandPool);
    LayersCombiner.freeCommandBuffer(commandPool);
    Shadow.freeCommandBuffer(commandPool);
    Skybox.freeCommandBuffer(commandPool);
    SSAO.freeCommandBuffer(commandPool);
    SSLR.freeCommandBuffer(commandPool);
    DeferredGraphics.freeCommandBuffer(commandPool);
    for(auto& layer: TransparentLayers){
        layer.freeCommandBuffer(commandPool);
    }

    for(auto& node: nodes){
        node->destroy(device.getLogical());
        delete node;
    }
    nodes.clear();
}

void deferredGraphics::destroyGraphics()
{
    freeCommandBuffers();

    DeferredGraphics.destroy();
    Filter.destroy();
    SSAO.destroy();
    SSLR.destroy();
    Skybox.destroy();
    Shadow.destroy();
    PostProcessing.destroy();
    PostProcessing.destroySwapChainAttachments();
    LayersCombiner.destroy();
    Blur.destroy();
    for(auto& layer: TransparentLayers){
        layer.destroy();
    }

    blurAttachment.deleteAttachment(device.getLogical());
    blurAttachment.deleteSampler(device.getLogical());

    for(auto& attachment: blitAttachments){
        attachment.deleteAttachment(device.getLogical());
        attachment.deleteSampler(device.getLogical());
    }

    ssaoAttachment.deleteAttachment(device.getLogical());
    ssaoAttachment.deleteSampler(device.getLogical());

    sslrAttachment.deleteAttachment(device.getLogical());
    sslrAttachment.deleteSampler(device.getLogical());

    skyboxAttachment.deleteAttachment(device.getLogical());
    skyboxAttachment.deleteSampler(device.getLogical());

    for(auto& attachment: layersCombinedAttachment){
        attachment.deleteAttachment(device.getLogical());
        attachment.deleteSampler(device.getLogical());
    }

    deferredAttachments.deleteAttachment(device.getLogical());
    deferredAttachments.deleteSampler(device.getLogical());
    for(auto& attachment: transparentLayersAttachments){
        attachment.deleteAttachment(device.getLogical());
        attachment.deleteSampler(device.getLogical());
    }

    for (auto& buffer: storageBuffersHost){
        if(buffer.instance) vkDestroyBuffer(device.getLogical(), buffer.instance, nullptr);
        if(buffer.memory)   vkFreeMemory(device.getLogical(), buffer.memory, nullptr);
    }
    storageBuffersHost.clear();

    if(swapChain) {vkDestroySwapchainKHR(device.getLogical(), swapChain, nullptr); swapChain = VK_NULL_HANDLE;}
}

void deferredGraphics::destroyCommandPool()
{
    if(commandPool) {vkDestroyCommandPool(device.getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;}
}

void deferredGraphics::setDevices(uint32_t devicesCount, physicalDevice* devices)
{
    for(uint32_t i=0;i<devicesCount;i++){
        this->devices.push_back(devices[i]);
    }
    device = this->devices[0];

    DeferredGraphics.setDeviceProp(device.instance, device.getLogical());
    PostProcessing.setDeviceProp(device.instance, device.getLogical());
    Filter.setDeviceProp(device.instance, device.getLogical());
    SSLR.setDeviceProp(device.instance, device.getLogical());
    SSAO.setDeviceProp(device.instance, device.getLogical());
    Skybox.setDeviceProp(device.instance, device.getLogical());
    Shadow.setDeviceProp(device.instance, device.getLogical());
    LayersCombiner.setDeviceProp(device.instance, device.getLogical());
    Blur.setDeviceProp(device.instance, device.getLogical());
    for(auto& layer: TransparentLayers){
        layer.setDeviceProp(device.instance, device.getLogical());
    }
}

void deferredGraphics::setSupportImageCount(VkSurfaceKHR* surface)
{
    auto capabilities = SwapChain::queryingSupport(device.instance, *surface).capabilities;
    imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount){
        imageCount = capabilities.maxImageCount;
    }

    updateCommandBufferFlags.resize(imageCount, true);
}

void deferredGraphics::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);
}

namespace {
    void fastCreateFilterGraphics(filterGraphics* filter, uint32_t attachmentsNumber, attachments* attachments)
    {
        filter->setAttachments(attachmentsNumber,attachments);
        filter->createRenderPass();
        filter->createFramebuffers();
        filter->createPipelines();
        filter->createDescriptorPool();
        filter->createDescriptorSets();
    }

    void fastCreateGraphics(graphics* graphics, DeferredAttachments* attachments)
    {
        graphics->setAttachments(attachments);
        graphics->createAttachments(attachments);
        graphics->createRenderPass();
        graphics->createFramebuffers();
        graphics->createPipelines();
        graphics->createDescriptorPool();
        graphics->createDescriptorSets();
    }
}

void deferredGraphics::createGraphics(GLFWwindow* window, VkSurfaceKHR* surface)
{
    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(device.instance,*surface);

    if(extent.height==0&&extent.width==0){
        extent = SwapChain::queryingExtent(window, swapChainSupport.capabilities);
    }

    imageInfo shadowsInfo{imageCount,VK_FORMAT_D32_SFLOAT,VkExtent2D{1024,1024},MSAASamples};
    Shadow.setImageProp(&shadowsInfo);

    imageInfo info{imageCount, SwapChain::queryingSurfaceFormat(swapChainSupport.formats).format, extent, MSAASamples};
    DeferredGraphics.setImageProp(&info);
    Blur.setImageProp(&info);
    LayersCombiner.setImageProp(&info);
    Filter.setImageProp(&info);
    SSLR.setImageProp(&info);
    SSAO.setImageProp(&info);
    Skybox.setImageProp(&info);
    PostProcessing.setImageProp(&info);
    for(auto& layer: TransparentLayers){
        layer.setImageProp(&info);
    }

    if(enableSkybox){
        Skybox.createAttachments(1,&skyboxAttachment);
        fastCreateFilterGraphics(&Skybox,1,&skyboxAttachment);
        Skybox.updateDescriptorSets(cameraObject);
    }

    fastCreateGraphics(&DeferredGraphics, &deferredAttachments);

    if(enableTransparentLayers){
        transparentLayersAttachments.resize(TransparentLayersCount);
        for(uint32_t i=0;i<transparentLayersAttachments.size();i++){
            TransparentLayers[i].setTransparencyPass(true);
            TransparentLayers[i].setScattering(false);
            fastCreateGraphics(&TransparentLayers[i], &transparentLayersAttachments[i]);
        }

        layersCombinedAttachment.resize(2);
        LayersCombiner.setTransparentLayersCount(TransparentLayersCount);
        LayersCombiner.createAttachments(static_cast<uint32_t>(layersCombinedAttachment.size()),layersCombinedAttachment.data());
        fastCreateFilterGraphics(&LayersCombiner,static_cast<uint32_t>(layersCombinedAttachment.size()),layersCombinedAttachment.data());
        LayersCombiner.updateDescriptorSets(deferredAttachments,transparentLayersAttachments.data(),&skyboxAttachment,cameraObject);
    }

    if(enableBloom){
        blitAttachments.resize(blitAttachmentCount);
        Filter.createBufferAttachments();
        Filter.setBlitFactor(blitFactor);
        Filter.setSrcAttachment(enableTransparentLayers ? &layersCombinedAttachment[1] : &deferredAttachments.bloom);
        Filter.createAttachments(blitAttachmentCount,blitAttachments.data());
        fastCreateFilterGraphics(&Filter,blitAttachmentCount,blitAttachments.data());
        Filter.updateDescriptorSets();
        PostProcessing.setBlitAttachments(blitAttachmentCount,blitAttachments.data(),blitFactor);
    }else{
        PostProcessing.setBlitAttachments(blitAttachmentCount,nullptr,blitFactor);
    }
    if(enableBlur){
        Blur.createBufferAttachments();
        Blur.createAttachments(1,&blurAttachment);
        fastCreateFilterGraphics(&Blur,1,&blurAttachment);
        Blur.updateDescriptorSets(&deferredAttachments.blur);
        PostProcessing.setBlurAttachment(&blurAttachment);
    }
    if(enableSSAO){
        SSAO.createAttachments(1,&ssaoAttachment);
        fastCreateFilterGraphics(&SSAO,1,&ssaoAttachment);
        SSAO.updateDescriptorSets(cameraObject, deferredAttachments);
        PostProcessing.setSSAOAttachment(&ssaoAttachment);
    }
    if(enableSSLR){
        SSLR.createAttachments(1,&sslrAttachment);
        fastCreateFilterGraphics(&SSLR,1,&sslrAttachment);
        SSLR.updateDescriptorSets(cameraObject,deferredAttachments, enableTransparentLayers ? transparentLayersAttachments[0] : deferredAttachments);
        PostProcessing.setSSLRAttachment(&sslrAttachment);
    }

    Shadow.createRenderPass();
    Shadow.createPipelines();

    std::vector<uint32_t> queueIndices = {0};
    PostProcessing.setLayersAttachment(enableTransparentLayers ? &layersCombinedAttachment[0] : &deferredAttachments.image);
    PostProcessing.createSwapChain(&swapChain, window, &swapChainSupport, surface, static_cast<uint32_t>(queueIndices.size()), queueIndices.data());
    PostProcessing.createSwapChainAttachments(&swapChain);
    PostProcessing.createRenderPass();
    PostProcessing.createFramebuffers();
    PostProcessing.createPipelines();
    PostProcessing.createDescriptorPool();
    PostProcessing.createDescriptorSets();
    PostProcessing.updateDescriptorSets();

    createStorageBuffers(imageCount);
}

void deferredGraphics::createCommandBuffers()
{
    Shadow.createCommandBuffers(commandPool);
    Skybox.createCommandBuffers(commandPool);
    DeferredGraphics.createCommandBuffers(commandPool);
    for(auto& layer: TransparentLayers){
        layer.createCommandBuffers(commandPool);
    }
    Blur.createCommandBuffers(commandPool);
    SSLR.createCommandBuffers(commandPool);
    SSAO.createCommandBuffers(commandPool);
    LayersCombiner.createCommandBuffers(commandPool);
    Filter.createCommandBuffers(commandPool);
    PostProcessing.createCommandBuffers(commandPool);

    copyCommandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(imageCount);
    vkAllocateCommandBuffers(device.getLogical(), &allocInfo, copyCommandBuffers.data());

    updateCmdFlags();

    nodes.resize(imageCount);
    for(uint32_t imageIndex = 0; imageIndex < imageCount; imageIndex++){
        nodes[imageIndex]
         = new node({
            stage(  {copyCommandBuffers[imageIndex]},
                    {VK_PIPELINE_STAGE_TRANSFER_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {Shadow.getCommandBuffer(imageIndex)},
                    {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
                    device.getQueue(0,0)),
            stage(  {Skybox.getCommandBuffer(imageIndex)},
                    {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {DeferredGraphics.getCommandBuffer(imageIndex)},
                    {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                    device.getQueue(0,0)),
            stage(  getTransparentLayersCommandBuffers(imageIndex),
                    {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {LayersCombiner.getCommandBuffer(imageIndex)},
                    {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {SSLR.getCommandBuffer(imageIndex), SSAO.getCommandBuffer(imageIndex),
                     Filter.getCommandBuffer(imageIndex), Blur.getCommandBuffer(imageIndex),
                     PostProcessing.getCommandBuffer(imageIndex)},
                    {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
                    device.getQueue(0,0))
        }, nullptr)))));

        nodes[imageIndex]->createSemaphores(device.getLogical());
    }
}

std::vector<std::vector<VkSemaphore>> deferredGraphics::sibmit(std::vector<std::vector<VkSemaphore>>& externalSemaphore, std::vector<VkFence>& externalFence, uint32_t imageIndex)
{
    nodes[imageIndex]->setExternalSemaphore(externalSemaphore);
    nodes[imageIndex]->back()->setExternalFence(externalFence);

    nodes[imageIndex]->submit();

    return nodes[imageIndex]->back()->getBackSemaphores();
}

void deferredGraphics::updateDescriptorSets()
{
    std::vector<VkBuffer> storageBuffers;
    for(const auto& buffer: storageBuffersHost){
        storageBuffers.push_back(buffer.instance);
    }

    DeferredGraphics.updateDescriptorSets(nullptr, storageBuffers.data(), sizeof(StorageBufferObject), cameraObject);
    if(enableTransparentLayers){
        TransparentLayers[0].updateDescriptorSets(nullptr, storageBuffers.data(), sizeof(StorageBufferObject), cameraObject);
        for(uint32_t i=1;i<TransparentLayers.size();i++){
            TransparentLayers[i].updateDescriptorSets(&transparentLayersAttachments[i-1].depth, storageBuffers.data(), sizeof(StorageBufferObject), cameraObject);
        }
    }
}

void deferredGraphics::updateCommandBuffers()
{
    for(size_t imageIndex=0; imageIndex<imageCount; imageIndex++){
        updateCommandBuffer(imageIndex);
    }
}

void deferredGraphics::updateCommandBuffer(uint32_t imageIndex)
{
    if(updateCommandBufferFlags[imageIndex]){
        Shadow.beginCommandBuffer(imageIndex);
            Shadow.updateCommandBuffer(imageIndex);
        Shadow.endCommandBuffer(imageIndex);

        Skybox.beginCommandBuffer(imageIndex);
            if(enableSkybox){ Skybox.updateCommandBuffer(imageIndex);}
        Skybox.endCommandBuffer(imageIndex);

        DeferredGraphics.beginCommandBuffer(imageIndex);
            DeferredGraphics.updateCommandBuffer(imageIndex);
        DeferredGraphics.endCommandBuffer(imageIndex);

        for(auto& layer: TransparentLayers){
            layer.beginCommandBuffer(imageIndex);
                if(enableTransparentLayers){ layer.updateCommandBuffer(imageIndex);}
            layer.endCommandBuffer(imageIndex);
        }

        Blur.beginCommandBuffer(imageIndex);
            if(enableBlur){ Blur.updateCommandBuffer(imageIndex);}
        Blur.endCommandBuffer(imageIndex);

        SSLR.beginCommandBuffer(imageIndex);
            if(enableSSLR){ SSLR.updateCommandBuffer(imageIndex);}
        SSLR.endCommandBuffer(imageIndex);

        SSAO.beginCommandBuffer(imageIndex);
            if(enableSSAO){ SSAO.updateCommandBuffer(imageIndex);}
        SSAO.endCommandBuffer(imageIndex);

        LayersCombiner.beginCommandBuffer(imageIndex);
            if(enableTransparentLayers){
                LayersCombiner.updateCommandBuffer(imageIndex);
            } else {
                Texture::transitionLayout( LayersCombiner.getCommandBuffer(imageIndex),
                                           deferredAttachments.bloom.image[imageIndex],
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           VK_REMAINING_MIP_LEVELS,
                                           0,
                                           1);
            }
        LayersCombiner.endCommandBuffer(imageIndex);

        Filter.beginCommandBuffer(imageIndex);
            if(enableBloom){ Filter.updateCommandBuffer(imageIndex);}
        Filter.endCommandBuffer(imageIndex);

        PostProcessing.beginCommandBuffer(imageIndex);
            PostProcessing.updateCommandBuffer(imageIndex);
        PostProcessing.endCommandBuffer(imageIndex);

        updateCommandBufferFlags[imageIndex] = false;
    }
}

void deferredGraphics::updateBuffers(uint32_t imageIndex)
{
    vkResetCommandBuffer(copyCommandBuffers[imageIndex],0);

     VkCommandBufferBeginInfo beginInfo{};
         beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
         beginInfo.flags = 0;
         beginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(copyCommandBuffers[imageIndex], &beginInfo);

    cameraObject->updateUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);
    Skybox.updateObjectUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);
    DeferredGraphics.updateObjectUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);
    DeferredGraphics.updateLightSourcesUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);

    vkEndCommandBuffer(copyCommandBuffers[imageIndex]);
}

std::vector<VkCommandBuffer> deferredGraphics::getTransparentLayersCommandBuffers(uint32_t imageIndex)
{
    std::vector<VkCommandBuffer> commandBuffers;
    for(auto& transparentLayer: TransparentLayers){
        commandBuffers.push_back(transparentLayer.getCommandBuffer(imageIndex));
    }
    return commandBuffers;
}

void deferredGraphics::createStorageBuffers(uint32_t imageCount)
{
    storageBuffersHost.resize(imageCount);
    for (auto& buffer: storageBuffersHost){
        Buffer::create( device.instance,
                        device.getLogical(),
                        sizeof(StorageBufferObject),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
        vkMapMemory(device.getLogical(), buffer.memory, 0, sizeof(StorageBufferObject), 0, &buffer.map);
    }
}

void deferredGraphics::updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey){
    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = glm::vec4(mousex,mousey,0.0f,0.0f);
        StorageUBO.number = INT_FAST32_MAX;
        StorageUBO.depth = 1.0f;
    std::memcpy(storageBuffersHost[currentImage].map, &StorageUBO, sizeof(StorageUBO));
}

uint32_t deferredGraphics::readStorageBuffer(uint32_t currentImage){
    StorageBufferObject storageBuffer{};
    std::memcpy(&storageBuffer, storageBuffersHost[currentImage].map, sizeof(StorageBufferObject));
    return storageBuffer.number;
}

uint32_t deferredGraphics::getImageCount()       {   return imageCount;}
VkSwapchainKHR& deferredGraphics::getSwapChain() {   return swapChain;}

void deferredGraphics::setExtent(VkExtent2D extent)             {   this->extent = extent;}
void deferredGraphics::setExternalPath(const std::string &path) {   ExternalPath = path;}
void deferredGraphics::setEmptyTexture(std::string ZERO_TEXTURE){
    this->emptyTexture = new texture(ZERO_TEXTURE);

    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    emptyTexture->createTextureImage(device.instance, device.getLogical(), commandBuffer);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
    emptyTexture->destroyStagingBuffer(device.getLogical());

    emptyTexture->createTextureImageView(device.getLogical());
    emptyTexture->createTextureSampler(device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});

    DeferredGraphics.setEmptyTexture(emptyTexture);
    for(auto& layer: TransparentLayers){
        layer.setEmptyTexture(emptyTexture);
    }

    Blur.setEmptyTexture(emptyTexture);
    Filter.setEmptyTexture(emptyTexture);
    LayersCombiner.setEmptyTexture(emptyTexture);
    Skybox.setEmptyTexture(emptyTexture);
    SSAO.setEmptyTexture(emptyTexture);
    SSLR.setEmptyTexture(emptyTexture);
    Shadow.setEmptyTexture(emptyTexture);
    PostProcessing.setEmptyTexture(emptyTexture);
}

void deferredGraphics::createModel(model *pModel){
    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    pModel->loadFromFile(device.instance, device.getLogical(), commandBuffer);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0), commandPool, &commandBuffer);
    pModel->destroyStagingBuffer(device.getLogical());
    pModel->createDescriptorPool(device.getLogical());
    pModel->createDescriptorSet(device.getLogical(), emptyTexture);
}

void deferredGraphics::destroyModel(model* pModel){
    pModel->destroy(device.getLogical());
}

void deferredGraphics::bindCameraObject(camera* cameraObject){
    this->cameraObject = cameraObject;
    cameraObject->createUniformBuffers(device.instance,device.getLogical(),imageCount);
}

void deferredGraphics::removeCameraObject(camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject->destroy(device.getLogical());
        this->cameraObject = nullptr;
    }
}

void deferredGraphics::bindLightSource(light* lightSource){
    if(lightSource->getTexture()){
        VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
        lightSource->getTexture()->createTextureImage(device.instance, device.getLogical(), commandBuffer);
        SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
        lightSource->getTexture()->destroyStagingBuffer(device.getLogical());
        lightSource->getTexture()->createTextureImageView(device.getLogical());
        lightSource->getTexture()->createTextureSampler(device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    }
    if(lightSource->isShadowEnable()){
        Shadow.addLightSource(lightSource);
        Shadow.createAttachments(1,lightSource->getAttachments());
        Shadow.setAttachments(1,lightSource->getAttachments());
        Shadow.createFramebuffers(lightSource);
    }

    lightSource->createUniformBuffers(device.instance,device.getLogical(),imageCount);

    lightSource->createDescriptorPool(device.getLogical(), imageCount);
    lightSource->createDescriptorSets(device.getLogical(), imageCount);
    lightSource->updateDescriptorSets(device.getLogical(), imageCount, emptyTexture);

    DeferredGraphics.bindLightSource(lightSource);
    for(auto& TransparentLayer: TransparentLayers){
        TransparentLayer.bindLightSource(lightSource);
    }

    updateCmdFlags();
}

void deferredGraphics::removeLightSource(light* lightSource){
    if(lightSource->getAttachments()){
        lightSource->getAttachments()->deleteAttachment(device.getLogical());
        lightSource->getAttachments()->deleteSampler(device.getLogical());
    }
    lightSource->destroy(device.getLogical());

    DeferredGraphics.removeLightSource(lightSource);
    for(auto& TransparentLayer: TransparentLayers){
        TransparentLayer.removeLightSource(lightSource);
    }

    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(device.getLogical());
    }
    Shadow.removeLightSource(lightSource);

    updateCmdFlags();
}

void deferredGraphics::bindObject(object* object){
    if(object->getTexture() && (object->getPipelineBitMask() & (0x1))){
        VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
        object->getTexture()->createTextureImage(device.instance, device.getLogical(), commandBuffer);
        SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
        object->getTexture()->createTextureImageView(device.getLogical());
        object->getTexture()->createTextureSampler(device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        object->getTexture()->destroyStagingBuffer(device.getLogical());
    }
    object->createUniformBuffers(device.instance,device.getLogical(),imageCount);
    object->createDescriptorPool(device.getLogical(),imageCount);
    object->createDescriptorSet(device.getLogical(),imageCount);

    switch (object->getPipelineBitMask()) {
        case (0<<4)|0x0:
        case (1<<4)|0x0:
            Shadow.bindBaseObject(object);
            DeferredGraphics.bindBaseObject(object);
            for(auto& layer: TransparentLayers){
                layer.bindBaseObject(object);
            }
            break;
        case (0<<4)|0x1:
            Skybox.bindObject(object);
            break;
    }

    updateCmdFlags();
}

bool deferredGraphics::removeObject(object* object){
    object->destroy(device.getLogical());
    if(object->getTexture() && (object->getPipelineBitMask() & (0x1))){
        object->getTexture()->destroy(device.getLogical());
    }

    bool res = true;

    switch (object->getPipelineBitMask()) {
        case (0<<4)|0x0:
        case (1<<4)|0x0:
            res = res && Shadow.removeBaseObject(object) && DeferredGraphics.removeBaseObject(object);
            for(auto& layer: TransparentLayers){
                res = res && layer.removeBaseObject(object);
            }
            break;
        case (0<<4)|0x1:
            res = res && Skybox.removeObject(object);
            break;
    }

    updateCmdFlags();

    return res;
}

void deferredGraphics::setMinAmbientFactor(const float& minAmbientFactor){
    DeferredGraphics.setMinAmbientFactor(minAmbientFactor);
    for(auto& layer: TransparentLayers){
        layer.setMinAmbientFactor(minAmbientFactor);
    }

    updateCmdFlags();
}

void deferredGraphics::updateCmdFlags(){
    for(auto flag: updateCommandBufferFlags){
        flag = true;
    }
}
