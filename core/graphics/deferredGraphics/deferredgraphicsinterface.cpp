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
    for(uint32_t i=0;i<TransparentLayersCount;i++)
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
    DeferredGraphics.destroyEmptyTexture();

    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].destroyEmptyTexture();
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

    for(size_t i=0; i<blitAttachmentCount; i++){
        blitAttachments[i].deleteAttachment(devicesInfo[0].device);
        blitAttachments[i].deleteSampler(devicesInfo[0].device);
    }

    ssaoAttachment.deleteAttachment(devicesInfo[0].device);
    ssaoAttachment.deleteSampler(devicesInfo[0].device);

    sslrAttachment.deleteAttachment(devicesInfo[0].device);
    sslrAttachment.deleteSampler(devicesInfo[0].device);

    skyboxAttachment.deleteAttachment(devicesInfo[0].device);
    skyboxAttachment.deleteSampler(devicesInfo[0].device);

    for(auto attachment: layersCombinedAttachment){
        attachment.deleteAttachment(devicesInfo[0].device);
        attachment.deleteSampler(devicesInfo[0].device);
    }

    deferredAttachments.deleteAttachment(devicesInfo[0].device);
    deferredAttachments.deleteSampler(devicesInfo[0].device);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        transparentLayersAttachments[i].deleteAttachment(devicesInfo[0].device);
        transparentLayersAttachments[i].deleteSampler(devicesInfo[0].device);
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

    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
}

void deferredGraphicsInterface::setSupportImageCount(VkSurfaceKHR* surface)
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(*devicesInfo[0].physicalDevice,*surface);
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
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(*devicesInfo[0].physicalDevice,*surface);
    VkSurfaceFormatKHR      surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);

    if(extent.height==0&&extent.width==0){
        extent = chooseSwapExtent(window, swapChainSupport.capabilities);
    }

    if(MSAASamples != VK_SAMPLE_COUNT_1_BIT){
        VkSampleCountFlagBits maxMSAASamples = getMaxUsableSampleCount(*devicesInfo[0].physicalDevice);
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

    fastCreateGraphics(&DeferredGraphics, &deferredAttachments);

    transparentLayersAttachments.resize(TransparentLayersCount);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        TransparentLayers[i].setTransparencyPass(true);
        TransparentLayers[i].setScattering(false);
        fastCreateGraphics(&TransparentLayers[i], &transparentLayersAttachments[i]);
    }

    fastCreateFilterGraphics(&Skybox,1,&skyboxAttachment);
    Skybox.updateDescriptorSets(cameraObject);

    Blur.createBufferAttachments();
    fastCreateFilterGraphics(&Blur,1,&blurAttachment);
    Blur.updateDescriptorSets(&deferredAttachments.blur);

    layersCombinedAttachment.resize(2);
    LayersCombiner.setTransparentLayersCount(TransparentLayersCount);
    fastCreateFilterGraphics(&LayersCombiner,static_cast<uint32_t>(layersCombinedAttachment.size()),layersCombinedAttachment.data());
    LayersCombiner.updateDescriptorSets(deferredAttachments,transparentLayersAttachments.data(),&skyboxAttachment,cameraObject);

    blitAttachments.resize(blitAttachmentCount);
    Filter.createBufferAttachments();
    Filter.setBlitFactor(blitFactor);
    Filter.setSrcAttachment(&layersCombinedAttachment[1]);
    fastCreateFilterGraphics(&Filter,blitAttachmentCount,blitAttachments.data());
    Filter.updateDescriptorSets();

    fastCreateFilterGraphics(&SSAO,1,&ssaoAttachment);
    SSAO.updateDescriptorSets(cameraObject, deferredAttachments);

    fastCreateFilterGraphics(&SSLR,1,&sslrAttachment);
    SSLR.updateDescriptorSets(cameraObject,deferredAttachments,transparentLayersAttachments[0]);

    PostProcessing.setBlurAttachment(&blurAttachment);
    PostProcessing.setBlitAttachments(blitAttachmentCount,blitAttachments.data(),blitFactor);
    PostProcessing.setSSAOAttachment(&ssaoAttachment);
    PostProcessing.setSSLRAttachment(&sslrAttachment);
    PostProcessing.setLayersAttachment(&layersCombinedAttachment[0]);
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

    TransparentLayers[0].updateDescriptorSets(nullptr, storageBuffers.data(), cameraObject);
    for(uint32_t i=1;i<TransparentLayersCount;i++){
        TransparentLayers[i].updateDescriptorSets(&transparentLayersAttachments[i-1].depth, storageBuffers.data(), cameraObject);
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

        Skybox.render(imageIndex,*commandBuffer);

        DeferredGraphics.render(imageIndex,*commandBuffer);
        for(uint32_t i=0;i<TransparentLayersCount;i++)
            TransparentLayers[i].render(imageIndex,*commandBuffer);

        Blur.render(imageIndex,*commandBuffer);
        SSLR.render(imageIndex,*commandBuffer);
        SSAO.render(imageIndex,*commandBuffer);
        LayersCombiner.render(imageIndex,*commandBuffer);
        Filter.render(imageIndex,*commandBuffer);

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
    cameraObject->updateUniformBuffer(devicesInfo[0].device,imageIndex);
    Skybox.updateObjectUniformBuffer(imageIndex);
    DeferredGraphics.updateObjectUniformBuffer(imageIndex);
    DeferredGraphics.updateLightSourcesUniformBuffer(imageIndex);
}

void deferredGraphicsInterface::createStorageBuffers(uint32_t imageCount)
{
    storageBuffers.resize(imageCount);
    storageBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(   devicesInfo[0].physicalDevice,
                        devicesInfo[0].device,
                        sizeof(StorageBufferObject),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        storageBuffers[i],
                        storageBuffersMemory[i]);
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

void deferredGraphicsInterface::updateCmdFlags()
{
    for(uint32_t index = 0; index < imageCount; index++){
        updateCommandBufferFlags[index] = true;
    }
}

void deferredGraphicsInterface::setExtent(VkExtent2D extent)             {   this->extent = extent;}
void deferredGraphicsInterface::setExternalPath(const std::string &path) {   ExternalPath = path;}
void deferredGraphicsInterface::setEmptyTexture(std::string ZERO_TEXTURE){
    DeferredGraphics.setEmptyTexture(ZERO_TEXTURE);

    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setEmptyTexture(ZERO_TEXTURE);
}

void deferredGraphicsInterface::createModel(gltfModel *pModel){
    pModel->loadFromFile(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool,1.0f);
    pModel->createDescriptorPool(devicesInfo[0].device);
    pModel->createDescriptorSet(devicesInfo[0].device,DeferredGraphics.getEmptyTexture());
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
        lightSource->getTexture()->createTextureImage(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool);
        lightSource->getTexture()->createTextureImageView(devicesInfo[0].deviceInfo::device);
        lightSource->getTexture()->createTextureSampler(devicesInfo[0].device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    }
    lightSource->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);

    lightSource->createShadow(devicesInfo[0].physicalDevice,devicesInfo[0].device,&indices,imageCount,ExternalPath);
    lightSource->updateShadowDescriptorSets();
    lightSource->createShadowCommandBuffers();

    lightSource->createDescriptorPool(devicesInfo[0].device, imageCount);
    lightSource->createDescriptorSets(devicesInfo[0].device, imageCount);
    lightSource->updateDescriptorSets(devicesInfo[0].device, imageCount,DeferredGraphics.getEmptyTexture());

    DeferredGraphics.addLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].addLightSource(lightSource);

    for(uint32_t index = 0; index < imageCount; index++){
        updateCommandBufferFlags[index] = true;
        updateShadowCommandBufferFlags[index] = true;
    }
}

void deferredGraphicsInterface::removeLightSource(light* lightSource){
    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(devicesInfo[0].device);
    }
    lightSource->destroyUniformBuffers(devicesInfo[0].device);
    lightSource->destroy(devicesInfo[0].device);

    DeferredGraphics.removeLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].removeLightSource(lightSource);

    for(uint32_t index = 0; index < imageCount; index++){
        updateCommandBufferFlags[index] = true;
        updateShadowCommandBufferFlags[index] = true;
    }
}

void deferredGraphicsInterface::bindBaseObject(object* object){
    object->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    object->createDescriptorPool(devicesInfo[0].device,imageCount);
    object->createDescriptorSet(devicesInfo[0].device,imageCount);

    DeferredGraphics.bindBaseObject(object);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].bindBaseObject(object);

    for(uint32_t index = 0; index < imageCount; index++){
        updateCommandBufferFlags[index] = true;
        updateShadowCommandBufferFlags[index] = true;
    }
}

bool deferredGraphicsInterface::removeObject(object* object){
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);

    bool res = true;
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        res = res&&(TransparentLayers[i].removeBaseObject(object));

    for(uint32_t index = 0; index < imageCount; index++){
        updateCommandBufferFlags[index] = true;
        updateShadowCommandBufferFlags[index] = true;
    }

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
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setMinAmbientFactor(minAmbientFactor);

    for(uint32_t index = 0; index < imageCount; index++){
        updateCommandBufferFlags[index] = true;
    }
}
