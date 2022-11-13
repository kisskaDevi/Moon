#include "deferredgraphicsinterface.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/lightInterface.h"
#include "core/transformational/object.h"

#include <iostream>

deferredGraphicsInterface::deferredGraphicsInterface(const std::string& ExternalPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples):
    ExternalPath(ExternalPath), extent(extent), MSAASamples(MSAASamples)
{
    DeferredGraphics.setExternalPath(ExternalPath);
    PostProcessing.setExternalPath(ExternalPath);
    Filter.setExternalPath(ExternalPath);
    SSLR.setExternalPath(ExternalPath);
    SSAO.setExternalPath(ExternalPath);
    Combiner.setExternalPath(ExternalPath);
    Blur.setExternalPath(ExternalPath);

    TransparentLayers.resize(TransparentLayersCount);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setExternalPath(ExternalPath);
}

deferredGraphicsInterface::~deferredGraphicsInterface()
{}

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
    PostProcessing.destroy();
    Combiner.destroy();
    Blur.destroy();

    blurAttachment.deleteAttachment(devicesInfo[0].device);
    blurAttachment.deleteSampler(devicesInfo[0].device);

    combineBloomAttachment.deleteAttachment(devicesInfo[0].device);
    combineBloomAttachment.deleteSampler(devicesInfo[0].device);

    for(size_t i=0; i<blitAttachmentCount; i++){
        blitAttachments[i].deleteAttachment(devicesInfo[0].device);
        blitAttachments[i].deleteSampler(devicesInfo[0].device);
    }

    ssaoAttachment.deleteAttachment(devicesInfo[0].device);
    ssaoAttachment.deleteSampler(devicesInfo[0].device);

    sslrAttachment.deleteAttachment(devicesInfo[0].device);
    sslrAttachment.deleteSampler(devicesInfo[0].device);

    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].destroy();
}

uint32_t deferredGraphicsInterface::getImageCount()
{
    return imageCount;
}

VkSwapchainKHR& deferredGraphicsInterface::getSwapChain()
{
    return swapChain;
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
    Combiner.setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
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
}

void deferredGraphicsInterface::createGraphics(GLFWwindow* window, VkSurfaceKHR* surface)
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(*devicesInfo[0].physicalDevice,*surface);
    VkSurfaceFormatKHR      surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);

    if(extent.height==0&&extent.width==0)
        extent = chooseSwapExtent(window, swapChainSupport.capabilities);

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
    Combiner.setImageProp(&info);
    Filter.setImageProp(&info);
    SSLR.setImageProp(&info);
    SSAO.setImageProp(&info);
    PostProcessing.setImageProp(&info);

    DeferredGraphics.createAttachments(&deferredAttachments);
    DeferredGraphics.createRenderPass();
    DeferredGraphics.createFramebuffers();
    DeferredGraphics.createPipelines();
    DeferredGraphics.createStorageBuffers(imageCount);
    DeferredGraphics.createBaseDescriptorPool();
    DeferredGraphics.createBaseDescriptorSets();
    DeferredGraphics.createSkyboxDescriptorPool();
    DeferredGraphics.createSkyboxDescriptorSets();
    DeferredGraphics.createLightingDescriptorPool();
    DeferredGraphics.createLightingDescriptorSets();

    transparentLayersAttachments.resize(TransparentLayersCount);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        TransparentLayers[i].setImageProp(&info);
        TransparentLayers[i].setTransparencyPass(true);
        TransparentLayers[i].setScattering(false);
        TransparentLayers[i].createAttachments(&transparentLayersAttachments[i]);
        TransparentLayers[i].createRenderPass();
        TransparentLayers[i].createFramebuffers();
        TransparentLayers[i].createPipelines();
        TransparentLayers[i].createStorageBuffers(imageCount);
        TransparentLayers[i].createBaseDescriptorPool();
        TransparentLayers[i].createBaseDescriptorSets();
        TransparentLayers[i].createSkyboxDescriptorPool();
        TransparentLayers[i].createSkyboxDescriptorSets();
        TransparentLayers[i].createLightingDescriptorPool();
        TransparentLayers[i].createLightingDescriptorSets();
    }

    std::vector<attachments> bloomAttachments(TransparentLayersCount+1);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        bloomAttachments[i] = transparentLayersAttachments[i].bloom;
    }
    bloomAttachments[TransparentLayersCount] = deferredAttachments.bloom;

    std::vector<attachment> depthAttachments(TransparentLayersCount+1);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        depthAttachments[i] = transparentLayersAttachments[i].depth;
    }
    depthAttachments[TransparentLayersCount] = deferredAttachments.depth;

    Blur.createBufferAttachments();
    Blur.setAttachments(1,&blurAttachment);
    Blur.createAttachments(1,&blurAttachment);
    Blur.createRenderPass();
    Blur.createFramebuffers();
    Blur.createPipelines();
    Blur.createDescriptorPool();
    Blur.createDescriptorSets();
    Blur.updateDescriptorSets(&deferredAttachments.blur);

    Combiner.setCombineAttachmentsCount(TransparentLayersCount+1);
    Combiner.setAttachments(1,&combineBloomAttachment);
    Combiner.createAttachments(1,&combineBloomAttachment);
    Combiner.createRenderPass();
    Combiner.createFramebuffers();
    Combiner.createPipelines();
    Combiner.createDescriptorPool();
    Combiner.createDescriptorSets();
    Combiner.updateDescriptorSets(bloomAttachments.data(),depthAttachments.data(),&deferredAttachments.depth);

    blitAttachments.resize(blitAttachmentCount);
    Filter.createBufferAttachments();
    Filter.setBlitFactor(blitFactor);
    Filter.setSrcAttachment(&combineBloomAttachment);
    Filter.setAttachments(blitAttachmentCount,blitAttachments.data());
    Filter.createAttachments(blitAttachmentCount,blitAttachments.data());
    Filter.createRenderPass();
    Filter.createFramebuffers();
    Filter.createPipelines();
    Filter.createDescriptorPool();
    Filter.createDescriptorSets();
    Filter.updateDescriptorSets();

    SSAO.setAttachments(1,&ssaoAttachment);
    SSAO.createAttachments(1,&ssaoAttachment);
    SSAO.createRenderPass();
    SSAO.createFramebuffers();
    SSAO.createPipelines();
    SSAO.createDescriptorPool();
    SSAO.createDescriptorSets();
    SSAO.updateDescriptorSets(deferredAttachments,DeferredGraphics.getSceneBuffer().data());

    SSLR.setAttachments(1,&sslrAttachment);
    SSLR.createAttachments(1,&sslrAttachment);
    SSLR.createRenderPass();
    SSLR.createFramebuffers();
    SSLR.createPipelines();
    SSLR.createDescriptorPool();
    SSLR.createDescriptorSets();
    SSLR.updateDescriptorSets(deferredAttachments,DeferredGraphics.getSceneBuffer().data());

    std::vector<DeferredAttachments> transparentLayers(TransparentLayersCount);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        transparentLayers[i] = transparentLayersAttachments[i];

    PostProcessing.setSwapChain(&swapChain);
    PostProcessing.setBlurAttachment(&blurAttachment);
    PostProcessing.setBlitAttachments(blitAttachmentCount,blitAttachments.data(),blitFactor);
    PostProcessing.setSSAOAttachment(&ssaoAttachment);
    PostProcessing.setSSLRAttachment(&sslrAttachment);
    PostProcessing.setTransparentLayersCount(TransparentLayersCount);
    PostProcessing.createAttachments(window, swapChainSupport, surface);
    PostProcessing.createRenderPass();
    PostProcessing.createFramebuffers();
    PostProcessing.createPipelines();
    PostProcessing.createDescriptorPool();
    PostProcessing.createDescriptorSets();
    PostProcessing.updateDescriptorSets(deferredAttachments,transparentLayers);
}

void deferredGraphicsInterface::createCommandBuffers()
{
    commandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = *devicesInfo[0].commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();
    if (vkAllocateCommandBuffers(*devicesInfo[0].device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate command buffers!");
}

void deferredGraphicsInterface::updateDescriptorSets()
{
    DeferredGraphics.updateBaseDescriptorSets(nullptr);
    DeferredGraphics.updateSkyboxDescriptorSets();
    DeferredGraphics.updateLightingDescriptorSets();

    TransparentLayers[0].updateBaseDescriptorSets(nullptr);
    //TransparentLayers[0].updateSkyboxDescriptorSets();
    TransparentLayers[0].updateLightingDescriptorSets();
    for(uint32_t i=1;i<TransparentLayersCount;i++){
        TransparentLayers[i].updateBaseDescriptorSets(&transparentLayersAttachments[i-1].depth);
        //TransparentLayers[i].updateSkyboxDescriptorSets();
        TransparentLayers[i].updateLightingDescriptorSets();
    }
}

void deferredGraphicsInterface::updateAllCommandBuffers()
{
    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        updateCommandBuffer(imageIndex, &commandBuffers[imageIndex]);

    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        DeferredGraphics.updateLightCmd(imageIndex);

    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++){
        for(uint32_t i=0;i<TransparentLayersCount;i++){
            TransparentLayers[i].updateLightCmd(imageIndex);
        }
    }
}

void deferredGraphicsInterface::updateCommandBuffer(uint32_t imageIndex, VkCommandBuffer* commandBuffer)
{
    vkResetCommandBuffer(*commandBuffer,0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                                            
        beginInfo.pInheritanceInfo = nullptr;
    if (vkBeginCommandBuffer(*commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording command buffer!");

    DeferredGraphics.render(imageIndex,*commandBuffer);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].render(imageIndex,*commandBuffer);

    Blur.render(imageIndex,*commandBuffer);
    Combiner.render(imageIndex,*commandBuffer);
    SSLR.render(imageIndex,*commandBuffer);
    SSAO.render(imageIndex,*commandBuffer);
    Filter.render(imageIndex,*commandBuffer);

    PostProcessing.render(imageIndex,*commandBuffer);

    if (vkEndCommandBuffer(*commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
}

VkCommandBuffer* deferredGraphicsInterface::getCommandBuffers(uint32_t& commandBuffersCount, uint32_t imageIndex)
{
    commandBufferSet.clear();

    DeferredGraphics.getLightCommandbuffers(commandBufferSet,imageIndex);
    commandBufferSet.push_back(this->commandBuffers[imageIndex]);

    commandBuffersCount = commandBufferSet.size();
    return commandBufferSet.data();
}

void deferredGraphicsInterface::updateCommandBuffers(uint32_t imageIndex)
{
    if(worldCmd.enable)
    {
        updateCommandBuffer(imageIndex, &commandBuffers[imageIndex]);
        if((++worldCmd.frames)==imageCount)
            worldCmd.enable = false;
    }
    if(lightsCmd.enable)
    {
        DeferredGraphics.updateLightCmd(imageIndex);
        if((++lightsCmd.frames)==imageCount)
            lightsCmd.enable = false;
    }
}

void deferredGraphicsInterface::updateBuffers(uint32_t imageIndex)
{
    if(worldUbo.enable)
    {
        updateUniformBuffer(imageIndex);
        if((++worldUbo.frames)==imageCount)
            worldUbo.enable = false;
    }
    if(lightsUbo.enable)
    {
        DeferredGraphics.updateLightUbo(imageIndex);
        if((++lightsUbo.frames)==imageCount)
            lightsUbo.enable = false;
    }
}

void deferredGraphicsInterface::freeCommandBuffers()
{
    vkFreeCommandBuffers(*devicesInfo[0].device, *devicesInfo[0].commandPool, static_cast<uint32_t>(commandBuffers.size()),commandBuffers.data());
    commandBuffers.clear();
}

void deferredGraphicsInterface::updateUniformBuffer(uint32_t imageIndex)
{
    DeferredGraphics.updateUniformBuffer(imageIndex);
    DeferredGraphics.updateSkyboxUniformBuffer(imageIndex);
    DeferredGraphics.updateObjectUniformBuffer(imageIndex);

    for(uint32_t i=0;i<TransparentLayersCount;i++){
        TransparentLayers[i].updateUniformBuffer(imageIndex);
        //TransparentLayers[i].updateSkyboxUniformBuffer(imageIndex);
        TransparentLayers[i].updateObjectUniformBuffer(imageIndex);
    }
}

void                                deferredGraphicsInterface::resetCmdLight(){ lightsCmd.enable = true; lightsCmd.frames = 0;}
void                                deferredGraphicsInterface::resetCmdWorld(){ worldCmd.enable = true;  worldCmd.frames = 0;}
void                                deferredGraphicsInterface::resetUboLight(){ lightsUbo.enable = true; lightsUbo.frames = 0;}
void                                deferredGraphicsInterface::resetUboWorld(){ worldUbo.enable = true;  worldUbo.frames = 0;}

void                                deferredGraphicsInterface::setExtent(VkExtent2D extent)             {   this->extent = extent;}
void                                deferredGraphicsInterface::setExternalPath(const std::string &path) {   ExternalPath = path;}
void                                deferredGraphicsInterface::setEmptyTexture(std::string ZERO_TEXTURE){
    DeferredGraphics.setEmptyTexture(ZERO_TEXTURE);

    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setEmptyTexture(ZERO_TEXTURE);
}

void                                deferredGraphicsInterface::setCameraObject(camera* cameraObject){
    DeferredGraphics.setCameraObject(cameraObject);

    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setCameraObject(cameraObject);
}

void                                deferredGraphicsInterface::createModel(gltfModel *pModel){
    pModel->loadFromFile(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool,1.0f);
    pModel->createDescriptorPool(devicesInfo[0].device);
    pModel->createDescriptorSet(devicesInfo[0].device,DeferredGraphics.getEmptyTexture());
}

void                                deferredGraphicsInterface::destroyModel(gltfModel* pModel){
    pModel->destroy(devicesInfo[0].device);
}

void                                deferredGraphicsInterface::bindLightSource(light* lightSource)
{
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
}
void                                deferredGraphicsInterface::removeLightSource(light* lightSource)
{
    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(devicesInfo[0].device);
    }
    lightSource->destroyUniformBuffers(devicesInfo[0].device);
    lightSource->destroy(devicesInfo[0].device);

    DeferredGraphics.removeLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].removeLightSource(lightSource);
}

void                                deferredGraphicsInterface::bindBaseObject(object* newObject)
{
    newObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    newObject->createDescriptorPool(devicesInfo[0].device,imageCount);
    newObject->createDescriptorSet(devicesInfo[0].device,imageCount);

    DeferredGraphics.bindBaseObject(newObject);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].bindBaseObject(newObject);
}
void                                deferredGraphicsInterface::bindOutliningObject(object* newObject, float lineWidth, glm::vec4 lineColor)
{
    newObject->setOutliningEnable(false);
    newObject->setOutliningWidth(lineWidth);
    newObject->setOutliningColor(lineColor);
    newObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    newObject->createDescriptorPool(devicesInfo[0].device,imageCount);
    newObject->createDescriptorSet(devicesInfo[0].device,imageCount);
    DeferredGraphics.bindOutliningObject(newObject);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].bindOutliningObject(newObject);
}
void                                deferredGraphicsInterface::bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH)
{
    DeferredGraphics.bindSkyBoxObject(newObject,TEXTURE_PATH);
}

bool                                deferredGraphicsInterface::removeBaseObject(object* object)
{
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);

    return DeferredGraphics.removeBaseObject(object);
}
bool                                deferredGraphicsInterface::removeOutliningObject(object* object)
{
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);

    return DeferredGraphics.removeOutliningObject(object);
}
bool                                deferredGraphicsInterface::removeSkyBoxObject(object* object)
{
    return DeferredGraphics.removeSkyBoxObject(object);
}

void                                deferredGraphicsInterface::removeBinds(){
    DeferredGraphics.removeBinds();
}

void                                deferredGraphicsInterface::setMinAmbientFactor(const float& minAmbientFactor){
    DeferredGraphics.setMinAmbientFactor(minAmbientFactor);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setMinAmbientFactor(minAmbientFactor);
}

void                                deferredGraphicsInterface::updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition){
    DeferredGraphics.updateStorageBuffer(currentImage,mousePosition);
}
uint32_t                            deferredGraphicsInterface::readStorageBuffer(uint32_t currentImage){
    return DeferredGraphics.readStorageBuffer(currentImage);
}
