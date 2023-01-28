#include "deferredgraphicsinterface.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/lightInterface.h"
#include "core/transformational/object.h"
#include "core/transformational/camera.h"
#include "bufferObjects.h"

#include <iostream>

deferredGraphicsInterface::deferredGraphicsInterface(const std::string& ExternalPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples):
    ExternalPath(ExternalPath), extent(extent), MSAASamples(MSAASamples)
{
    DeferredGraphics.setExternalPath(ExternalPath);
    PostProcessing.setExternalPath(ExternalPath);
    Filter.setExternalPath(ExternalPath);
    SSLR.setExternalPath(ExternalPath);
    SSAO.setExternalPath(ExternalPath);
    Skybox.setExternalPath(ExternalPath);
    LayersCombiner.setExternalPath(ExternalPath);
    Blur.setExternalPath(ExternalPath);

    TransparentLayers.resize(TransparentLayersCount);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].setExternalPath(ExternalPath);
}

deferredGraphicsInterface::~deferredGraphicsInterface()
{}

void deferredGraphicsInterface::freeCommandBuffers()
{
    if(commandBuffers.data()){
        vkFreeCommandBuffers(*devicesInfo[0].device, *devicesInfo[0].commandPool, static_cast<uint32_t>(commandBuffers.size()),commandBuffers.data());
    }
    commandBuffers.resize(0);
}

void deferredGraphicsInterface::destroyEmptyTextures()
{
    emptyTexture->destroy(devicesInfo[0].device);
    emptyTexture = nullptr;
}

void deferredGraphicsInterface::destroyGraphics()
{
    DeferredGraphics.destroy();
    Filter.destroy();
    SSAO.destroy();
    SSLR.destroy();
    Skybox.destroy();
    PostProcessing.destroy();
    PostProcessing.destroySwapChainAttachments();
    LayersCombiner.destroy();
    Blur.destroy();
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].destroy();

    blurAttachment.deleteAttachment(devicesInfo[0].device);
    blurAttachment.deleteSampler(devicesInfo[0].device);

    for(auto& attachment: blitAttachments){
        attachment.deleteAttachment(devicesInfo[0].device);
        attachment.deleteSampler(devicesInfo[0].device);
    }

    ssaoAttachment.deleteAttachment(devicesInfo[0].device);
    ssaoAttachment.deleteSampler(devicesInfo[0].device);

    sslrAttachment.deleteAttachment(devicesInfo[0].device);
    sslrAttachment.deleteSampler(devicesInfo[0].device);

    skyboxAttachment.deleteAttachment(devicesInfo[0].device);
    skyboxAttachment.deleteSampler(devicesInfo[0].device);

    for(auto& attachment: layersCombinedAttachment){
        attachment.deleteAttachment(devicesInfo[0].device);
        attachment.deleteSampler(devicesInfo[0].device);
    }

    deferredAttachments.deleteAttachment(devicesInfo[0].device);
    deferredAttachments.deleteSampler(devicesInfo[0].device);
    for(auto& attachment: transparentLayersAttachments){
        attachment.deleteAttachment(devicesInfo[0].device);
        attachment.deleteSampler(devicesInfo[0].device);
    }

    for (size_t i = 0; i < storageBuffers.size(); i++){
        if(storageBuffers[i])       vkDestroyBuffer(*devicesInfo[0].device, storageBuffers[i], nullptr);
        if(storageBuffersMemory[i]) vkFreeMemory(*devicesInfo[0].device, storageBuffersMemory[i], nullptr);
    }
    storageBuffers.resize(0);

    if(swapChain) {vkDestroySwapchainKHR(*devicesInfo[0].device, swapChain, nullptr); swapChain = VK_NULL_HANDLE;}
}

void deferredGraphicsInterface::setDevicesInfo(uint32_t devicesInfoCount, deviceInfo* devicesInfo)
{
    this->devicesInfo.resize(devicesInfoCount);
    for(uint32_t i=0;i<devicesInfoCount;i++){
        this->devicesInfo[i] = devicesInfo[i];
    }

    DeferredGraphics.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
    PostProcessing.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool,this->devicesInfo[0].graphicsFamily->value(),this->devicesInfo[0].presentFamily->value());
    Filter.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
    SSLR.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
    SSAO.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
    Skybox.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
    LayersCombiner.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
    Blur.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);

    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
}

void deferredGraphicsInterface::setSupportImageCount(VkSurfaceKHR* surface)
{
    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(*devicesInfo[0].physicalDevice,*surface);
    imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    updateCommandBufferFlags.resize(imageCount,true);
    updateShadowCommandBufferFlags.resize(imageCount,true);
}

void deferredGraphicsInterface::fastCreateFilterGraphics(filterGraphics* filter, uint32_t attachmentsNumber, attachments* attachments)
{
    filter->setAttachments(attachmentsNumber,attachments);
    filter->createAttachments(attachmentsNumber,attachments);
    filter->createRenderPass();
    filter->createFramebuffers();
    filter->createPipelines();
    filter->createDescriptorPool();
    filter->createDescriptorSets();
}

void deferredGraphicsInterface::fastCreateGraphics(deferredGraphics* graphics, DeferredAttachments* attachments)
{
    graphics->setAttachments(attachments);
    graphics->createAttachments(attachments);
    graphics->createRenderPass();
    graphics->createFramebuffers();
    graphics->createPipelines();
    graphics->createDescriptorPool();
    graphics->createDescriptorSets();
}

void deferredGraphicsInterface::createGraphics(GLFWwindow* window, VkSurfaceKHR* surface)
{
    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(*devicesInfo[0].physicalDevice,*surface);
    VkSurfaceFormatKHR      surfaceFormat = SwapChain::queryingSurfaceFormat(swapChainSupport.formats);

    if(extent.height==0&&extent.width==0){
        extent = SwapChain::queryingExtent(window, swapChainSupport.capabilities);
    }

    if(MSAASamples != VK_SAMPLE_COUNT_1_BIT){
        VkSampleCountFlagBits maxMSAASamples = PhysicalDevice::queryingSampleCount(*devicesInfo[0].physicalDevice);
        if(MSAASamples>maxMSAASamples)  MSAASamples = maxMSAASamples;
    }

    imageInfo info{};
        info.Count = imageCount;
        info.Format = surfaceFormat.format;
        info.Extent = extent;
        info.Samples = MSAASamples;
    DeferredGraphics.setImageProp(&info);
    Blur.setImageProp(&info);
    LayersCombiner.setImageProp(&info);
    Filter.setImageProp(&info);
    SSLR.setImageProp(&info);
    SSAO.setImageProp(&info);
    Skybox.setImageProp(&info);
    PostProcessing.setImageProp(&info);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        TransparentLayers[i].setImageProp(&info);
    }

    if(enableSkybox){
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
        fastCreateFilterGraphics(&LayersCombiner,static_cast<uint32_t>(layersCombinedAttachment.size()),layersCombinedAttachment.data());
        LayersCombiner.updateDescriptorSets(deferredAttachments,transparentLayersAttachments.data(),&skyboxAttachment,cameraObject);
    }

    if(enableBloom){
        blitAttachments.resize(blitAttachmentCount);
        Filter.createBufferAttachments();
        Filter.setBlitFactor(blitFactor);
        Filter.setSrcAttachment(enableTransparentLayers ? &layersCombinedAttachment[1] : &deferredAttachments.bloom);
        fastCreateFilterGraphics(&Filter,blitAttachmentCount,blitAttachments.data());
        Filter.updateDescriptorSets();
        PostProcessing.setBlitAttachments(blitAttachmentCount,blitAttachments.data(),blitFactor);
    }else{
        PostProcessing.setBlitAttachments(blitAttachmentCount,nullptr,blitFactor);
    }
    if(enableBlur){
        Blur.createBufferAttachments();
        fastCreateFilterGraphics(&Blur,1,&blurAttachment);
        Blur.updateDescriptorSets(&deferredAttachments.blur);
        PostProcessing.setBlurAttachment(&blurAttachment);
    }
    if(enableSSAO){
        fastCreateFilterGraphics(&SSAO,1,&ssaoAttachment);
        SSAO.updateDescriptorSets(cameraObject, deferredAttachments);
        PostProcessing.setSSAOAttachment(&ssaoAttachment);
    }
    if(enableSSLR){
        fastCreateFilterGraphics(&SSLR,1,&sslrAttachment);
        SSLR.updateDescriptorSets(cameraObject,deferredAttachments, enableTransparentLayers ? transparentLayersAttachments[0] : deferredAttachments);
        PostProcessing.setSSLRAttachment(&sslrAttachment);
    }

    PostProcessing.setLayersAttachment(enableTransparentLayers ? &layersCombinedAttachment[0] : &deferredAttachments.image);
    PostProcessing.createSwapChain(&swapChain, window, swapChainSupport, surface);
    PostProcessing.createSwapChainAttachments(&swapChain);
    PostProcessing.createRenderPass();
    PostProcessing.createFramebuffers();
    PostProcessing.createPipelines();
    PostProcessing.createDescriptorPool();
    PostProcessing.createDescriptorSets();
    PostProcessing.updateDescriptorSets();

    createStorageBuffers(imageCount);
}

void deferredGraphicsInterface::createCommandBuffers()
{
    commandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = *devicesInfo[0].commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    vkAllocateCommandBuffers(*devicesInfo[0].device, &allocInfo, commandBuffers.data());
}

void deferredGraphicsInterface::updateDescriptorSets()
{
    DeferredGraphics.updateDescriptorSets(nullptr, storageBuffers.data(), cameraObject);

    if(enableTransparentLayers){
        TransparentLayers[0].updateDescriptorSets(nullptr, storageBuffers.data(), cameraObject);
        for(uint32_t i=1;i<TransparentLayers.size();i++){
            TransparentLayers[i].updateDescriptorSets(&transparentLayersAttachments[i-1].depth, storageBuffers.data(), cameraObject);
        }
    }
}

void deferredGraphicsInterface::updateCommandBuffers()
{
    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        updateCommandBuffer(imageIndex, &commandBuffers[imageIndex]);
}

void deferredGraphicsInterface::updateCommandBuffer(uint32_t imageIndex, VkCommandBuffer* commandBuffer)
{
    vkResetCommandBuffer(*commandBuffer,0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;
    vkBeginCommandBuffer(*commandBuffer, &beginInfo);

        if(enableSkybox)            Skybox.render(imageIndex,*commandBuffer);

        DeferredGraphics.render(imageIndex,*commandBuffer);

        if(enableTransparentLayers)
            for(auto& layer: TransparentLayers)
               layer.render(imageIndex,*commandBuffer);

        if(enableBlur)              Blur.render(imageIndex,*commandBuffer);
        if(enableSSLR)              SSLR.render(imageIndex,*commandBuffer);
        if(enableSSAO)              SSAO.render(imageIndex,*commandBuffer);
        if(enableTransparentLayers){
                                    LayersCombiner.render(imageIndex,*commandBuffer);
        } else {
                                    Texture::transitionLayout( *commandBuffer,deferredAttachments.bloom.image[imageIndex],
                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                               VK_REMAINING_MIP_LEVELS,
                                                               0,
                                                               1);
        }
        if(enableBloom)             Filter.render(imageIndex,*commandBuffer);

        PostProcessing.render(imageIndex,*commandBuffer);

    vkEndCommandBuffer(*commandBuffer);
}

VkCommandBuffer* deferredGraphicsInterface::getCommandBuffers(uint32_t& commandBuffersCount, uint32_t imageIndex)
{
    commandBufferSet.clear();

    DeferredGraphics.getLightCommandbuffers(commandBufferSet,imageIndex);
    commandBufferSet.push_back(this->commandBuffers[imageIndex]);

    commandBuffersCount = commandBufferSet.size();
    return commandBufferSet.data();
}

void deferredGraphicsInterface::updateCommandBuffer(uint32_t imageIndex)
{
    if(updateCommandBufferFlags[imageIndex]){
        updateCommandBuffer(imageIndex, &commandBuffers[imageIndex]);
        updateCommandBufferFlags[imageIndex] = false;
    }
    if(updateShadowCommandBufferFlags[imageIndex]){
        DeferredGraphics.updateLightCmd(imageIndex);
        updateShadowCommandBufferFlags[imageIndex] = false;
    }
}

void deferredGraphicsInterface::updateBuffers(uint32_t imageIndex)
{
    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(*devicesInfo[0].device,*devicesInfo[0].commandPool);
    cameraObject->updateUniformBuffer(*devicesInfo[0].device, commandBuffer, imageIndex);
    Skybox.updateObjectUniformBuffer(commandBuffer, imageIndex);
    DeferredGraphics.updateObjectUniformBuffer(commandBuffer, imageIndex);
    DeferredGraphics.updateLightSourcesUniformBuffer(commandBuffer, imageIndex);
    SingleCommandBuffer::submit(*devicesInfo[0].device,*devicesInfo[0].queue,*devicesInfo[0].commandPool,&commandBuffer);
}

void deferredGraphicsInterface::createStorageBuffers(uint32_t imageCount)
{
    storageBuffers.resize(imageCount);
    storageBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        Buffer::create( *devicesInfo[0].physicalDevice,
                        *devicesInfo[0].device,
                        sizeof(StorageBufferObject),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &storageBuffers[i],
                        &storageBuffersMemory[i]);
    }
}

void deferredGraphicsInterface::updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey){
    void* data;

    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = glm::vec4(mousex,mousey,0.0f,0.0f);
        StorageUBO.number = INT_FAST32_MAX;
        StorageUBO.depth = 1.0f;
    vkMapMemory(*devicesInfo[0].device, storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(data, &StorageUBO, sizeof(StorageUBO));
    vkUnmapMemory(*devicesInfo[0].device, storageBuffersMemory[currentImage]);
}

uint32_t deferredGraphicsInterface::readStorageBuffer(uint32_t currentImage){
    void* data;

    StorageBufferObject StorageUBO{};
    vkMapMemory(*devicesInfo[0].device, storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(&StorageUBO, data, sizeof(StorageUBO));
    vkUnmapMemory(*devicesInfo[0].device, storageBuffersMemory[currentImage]);

    return StorageUBO.number;
}

uint32_t deferredGraphicsInterface::getImageCount()       {   return imageCount;}
VkSwapchainKHR& deferredGraphicsInterface::getSwapChain() {   return swapChain;}

void deferredGraphicsInterface::setExtent(VkExtent2D extent)             {   this->extent = extent;}
void deferredGraphicsInterface::setExternalPath(const std::string &path) {   ExternalPath = path;}
void deferredGraphicsInterface::setEmptyTexture(std::string ZERO_TEXTURE){
    this->emptyTexture = new texture(ZERO_TEXTURE);
    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(*devicesInfo[0].device,*devicesInfo[0].commandPool);
    emptyTexture->createTextureImage(*devicesInfo[0].physicalDevice, *devicesInfo[0].device, commandBuffer);
    SingleCommandBuffer::submit(*devicesInfo[0].device,*devicesInfo[0].queue,*devicesInfo[0].commandPool,&commandBuffer);
    emptyTexture->destroyStagingBuffer(devicesInfo[0].device);

    emptyTexture->createTextureImageView(devicesInfo[0].device);
    emptyTexture->createTextureSampler(devicesInfo[0].device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});

    DeferredGraphics.setEmptyTexture(emptyTexture);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].setEmptyTexture(emptyTexture);

    Blur.setEmptyTexture(emptyTexture);
    Filter.setEmptyTexture(emptyTexture);
    LayersCombiner.setEmptyTexture(emptyTexture);
    Skybox.setEmptyTexture(emptyTexture);
    SSAO.setEmptyTexture(emptyTexture);
    SSLR.setEmptyTexture(emptyTexture);
    PostProcessing.setEmptyTexture(emptyTexture);
}

void deferredGraphicsInterface::createModel(gltfModel *pModel){
    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(*devicesInfo[0].device,*devicesInfo[0].commandPool);
    pModel->loadFromFile(*devicesInfo[0].physicalDevice, *devicesInfo[0].device, commandBuffer);
    SingleCommandBuffer::submit(*devicesInfo[0].device, *devicesInfo[0].queue, *devicesInfo[0].commandPool, &commandBuffer);
    pModel->destroyStagingBuffer(*devicesInfo[0].device);
    pModel->createDescriptorPool(devicesInfo[0].device);
    pModel->createDescriptorSet(devicesInfo[0].device, emptyTexture);
}

void deferredGraphicsInterface::destroyModel(gltfModel* pModel){
    pModel->destroy(devicesInfo[0].device);
}

void deferredGraphicsInterface::bindCameraObject(camera* cameraObject){
    this->cameraObject = cameraObject;
    cameraObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
}

void deferredGraphicsInterface::removeCameraObject(camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject->destroyUniformBuffers(devicesInfo[0].device);
        this->cameraObject = nullptr;
    }
}

void deferredGraphicsInterface::bindLightSource(light* lightSource){
    QueueFamilyIndices indices{*devicesInfo[0].graphicsFamily,*devicesInfo[0].presentFamily};
    if(lightSource->getTexture()){
        VkCommandBuffer commandBuffer = SingleCommandBuffer::create(*devicesInfo[0].device,*devicesInfo[0].commandPool);
        lightSource->getTexture()->createTextureImage(*devicesInfo[0].physicalDevice, *devicesInfo[0].device, commandBuffer);
        SingleCommandBuffer::submit(*devicesInfo[0].device,*devicesInfo[0].queue,*devicesInfo[0].commandPool,&commandBuffer);
        lightSource->getTexture()->createTextureImageView(devicesInfo[0].deviceInfo::device);
        lightSource->getTexture()->createTextureSampler(devicesInfo[0].device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    }
    lightSource->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);

    lightSource->createShadow(devicesInfo[0].physicalDevice,devicesInfo[0].device,&indices,imageCount,ExternalPath);
    lightSource->updateShadowDescriptorSets();
    lightSource->createShadowCommandBuffers();

    lightSource->createDescriptorPool(devicesInfo[0].device, imageCount);
    lightSource->createDescriptorSets(devicesInfo[0].device, imageCount);
    lightSource->updateDescriptorSets(devicesInfo[0].device, imageCount, emptyTexture);

    DeferredGraphics.addLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].addLightSource(lightSource);

    updateCmdFlags();
    updateShadowCmdFlags();
}

void deferredGraphicsInterface::removeLightSource(light* lightSource){
    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(devicesInfo[0].device);
    }
    lightSource->destroyUniformBuffers(devicesInfo[0].device);
    lightSource->destroy(devicesInfo[0].device);

    DeferredGraphics.removeLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].removeLightSource(lightSource);

    updateCmdFlags();
    updateShadowCmdFlags();
}

void deferredGraphicsInterface::bindBaseObject(object* object){
    object->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    object->createDescriptorPool(devicesInfo[0].device,imageCount);
    object->createDescriptorSet(devicesInfo[0].device,imageCount);

    DeferredGraphics.bindBaseObject(object);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].bindBaseObject(object);

    updateCmdFlags();
    updateShadowCmdFlags();
}

bool deferredGraphicsInterface::removeObject(object* object){
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);

    bool res = true;
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        res = res&&(TransparentLayers[i].removeBaseObject(object));

    updateCmdFlags();
    updateShadowCmdFlags();

    return res&&(DeferredGraphics.removeBaseObject(object));
}

void deferredGraphicsInterface::bindSkyBoxObject(skyboxObject* object){
    object->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    object->createTexture(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool);
    object->createDescriptorPool(devicesInfo[0].device,imageCount);
    object->createDescriptorSet(devicesInfo[0].device,imageCount);

    Skybox.bindObject(object);
}

bool deferredGraphicsInterface::removeSkyBoxObject(skyboxObject* object){
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);
    object->destroyTexture(devicesInfo[0].device);

    return Skybox.removeObject(object);
}

void deferredGraphicsInterface::setMinAmbientFactor(const float& minAmbientFactor){
    DeferredGraphics.setMinAmbientFactor(minAmbientFactor);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].setMinAmbientFactor(minAmbientFactor);

    updateCmdFlags();
}

void deferredGraphicsInterface::updateCmdFlags(){
    for(uint32_t index = 0; index < imageCount; index++){
        updateCommandBufferFlags[index] = true;
    }
}

void deferredGraphicsInterface::updateShadowCmdFlags(){
    for(uint32_t index = 0; index < imageCount; index++){
        updateShadowCommandBufferFlags[index] = true;
    }
}
