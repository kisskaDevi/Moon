#include "deferredgraphicsinterface.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
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

    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setDeviceProp(this->devicesInfo[0].physicalDevice,this->devicesInfo[0].device,this->devicesInfo[0].queue,this->devicesInfo[0].commandPool);
}

void deferredGraphicsInterface::setSupportImageCount(VkSurfaceKHR* surface)
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(*devicesInfo[0].physicalDevice,*surface);                  //здест происходит запрос поддерживаемы режимов и форматов которые в следующий строчках передаются в соответствующие переменные через фукцнии
    imageCount = swapChainSupport.capabilities.minImageCount + 1;                                                               //запрос на поддержк уминимального количества числа изображений, число изображений равное 2 означает что один буфер передний, а второй задний
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)            //в первом условии мы проверяем доступно ли нам вообще какое-то количество изображений и проверяем не совпадает ли максимальное число изображений с минимальным
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

    blitAttachments.resize(blitAttachmentCount);
    PostProcessing.setSwapChain(&swapChain);
    PostProcessing.setBlitFactor(blitFactor);
    PostProcessing.setBlitAttachments(blitAttachmentCount,blitAttachments.data());
    PostProcessing.setBlitAttachment(&blitAttachment);
    PostProcessing.setSSAOAttachment(&ssaoAttachment);
    PostProcessing.setSSLRAttachment(&sslrAttachment);

    imageInfo info{};
        info.Count = imageCount;
        info.Format = surfaceFormat.format;
        info.Extent = extent;
        info.Samples = MSAASamples;
    PostProcessing.setImageProp(&info);
    DeferredGraphics.setImageProp(&info);
    Filter.setImageProp(&info);
    SSLR.setImageProp(&info);
    SSAO.setImageProp(&info);
    Combiner.setImageProp(&info);

    DeferredGraphics.createAttachments();
    DeferredGraphics.createRenderPass();
    DeferredGraphics.createFramebuffers();
    DeferredGraphics.createPipelines();
    DeferredGraphics.createStorageBuffers(imageCount);
    DeferredGraphics.createBaseDescriptorPool();
    DeferredGraphics.createBaseDescriptorSets();
    DeferredGraphics.createSkyboxDescriptorPool();
    DeferredGraphics.createSkyboxDescriptorSets();
    DeferredGraphics.createSpotLightingDescriptorPool();
    DeferredGraphics.createSpotLightingDescriptorSets();

    for(uint32_t i=0;i<TransparentLayersCount;i++){
        TransparentLayers[i].setImageProp(&info);
        TransparentLayers[i].setTransparencyPass(true);
        TransparentLayers[i].setScattering(false);
        TransparentLayers[i].createAttachments();
        TransparentLayers[i].createRenderPass();
        TransparentLayers[i].createFramebuffers();
        TransparentLayers[i].createPipelines();
        TransparentLayers[i].createStorageBuffers(imageCount);
        TransparentLayers[i].createBaseDescriptorPool();
        TransparentLayers[i].createBaseDescriptorSets();
        TransparentLayers[i].createSkyboxDescriptorPool();
        TransparentLayers[i].createSkyboxDescriptorSets();
        TransparentLayers[i].createSpotLightingDescriptorPool();
        TransparentLayers[i].createSpotLightingDescriptorSets();
    }

    std::vector<DeferredAttachments> transparentLayers(TransparentLayersCount);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        transparentLayers[i] = TransparentLayers[i].getDeferredAttachments();

    PostProcessing.setTransparentLayersCount(TransparentLayersCount);
    PostProcessing.createAttachments(window, swapChainSupport, surface);
    PostProcessing.createRenderPass();
    PostProcessing.createFramebuffers();
    PostProcessing.createPipelines();
    PostProcessing.createDescriptorPool();
    PostProcessing.createDescriptorSets(DeferredGraphics.getDeferredAttachments(),transparentLayers);

    std::vector<attachments> bloomAttachments(TransparentLayersCount+1);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        bloomAttachments[i] = *TransparentLayers[i].getDeferredAttachments().bloom;
    }
    bloomAttachments[TransparentLayersCount] = *DeferredGraphics.getDeferredAttachments().bloom;

    std::vector<attachment> depthAttachments(TransparentLayersCount+1);
    for(uint32_t i=0;i<TransparentLayersCount;i++){
        depthAttachments[i] = *TransparentLayers[i].getDeferredAttachments().depth;
    }
    depthAttachments[TransparentLayersCount] = *DeferredGraphics.getDeferredAttachments().depth;

    Combiner.setCombineAttachmentsCount(TransparentLayersCount+1);
    Combiner.setAttachments(&combineBloomAttachment);
    Combiner.createAttachments();
    Combiner.createRenderPass();
    Combiner.createFramebuffers();
    Combiner.createPipelines();
    Combiner.createDescriptorPool();
    Combiner.createDescriptorSets();
    Combiner.updateSecondDescriptorSets(bloomAttachments.data(),depthAttachments.data(),DeferredGraphics.getDeferredAttachments().depth);

    Filter.setBlitAttachments(&blitAttachment);
    Filter.setAttachments(blitAttachmentCount,blitAttachments.data());
    Filter.createRenderPass();
    Filter.createFramebuffers();
    Filter.createPipelines();
    Filter.createDescriptorPool();
    Filter.createDescriptorSets();
    Filter.updateSecondDescriptorSets();

    SSLR.setSSLRAttachments(&sslrAttachment);
    SSLR.createRenderPass();
    SSLR.createFramebuffers();
    SSLR.createPipelines();
    SSLR.createDescriptorPool();
    SSLR.createDescriptorSets();
    SSLR.updateSecondDescriptorSets(DeferredGraphics.getDeferredAttachments(),DeferredGraphics.getSceneBuffer().data());

    SSAO.setSSAOAttachments(&ssaoAttachment);
    SSAO.createRenderPass();
    SSAO.createFramebuffers();
    SSAO.createPipelines();
    SSAO.createDescriptorPool();
    SSAO.createDescriptorSets();
    SSAO.updateSecondDescriptorSets(DeferredGraphics.getDeferredAttachments(),DeferredGraphics.getSceneBuffer().data());
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
    DeferredGraphics.updateSpotLightingDescriptorSets();

    TransparentLayers[0].updateBaseDescriptorSets(nullptr);
    //TransparentLayers[0].updateSkyboxDescriptorSets();
    TransparentLayers[0].updateSpotLightingDescriptorSets();
    for(uint32_t i=1;i<TransparentLayersCount;i++){
        TransparentLayers[i].updateBaseDescriptorSets(TransparentLayers[i-1].getDeferredAttachments().depth);
        //TransparentLayers[i].updateSkyboxDescriptorSets();
        TransparentLayers[i].updateSpotLightingDescriptorSets();
    }
}

void deferredGraphicsInterface::updateAllCommandBuffers()
{
    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        updateCommandBuffer(imageIndex, &commandBuffers[imageIndex]);

    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        DeferredGraphics.updateSpotLightCmd(imageIndex);

    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++){
        for(uint32_t i=0;i<TransparentLayersCount;i++){
            TransparentLayers[i].updateSpotLightCmd(imageIndex);
        }
    }
}

void deferredGraphicsInterface::updateCommandBuffer(uint32_t imageIndex, VkCommandBuffer* commandBuffer)
{
    vkResetCommandBuffer(*commandBuffer,0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                                            //поле для передачи информации о том, как будет использоваться этот командный буфер (смотри страницу 102)
        beginInfo.pInheritanceInfo = nullptr;                           //используется при начале вторичного буфера, для того чтобы определить, какие состояния наследуются от первичного командного буфера, который его вызовет
    if (vkBeginCommandBuffer(*commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording command buffer!");

    DeferredGraphics.render(imageIndex,*commandBuffer);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].render(imageIndex,*commandBuffer);

    Combiner.render(imageIndex,*commandBuffer);
    SSLR.render(imageIndex,*commandBuffer);
    SSAO.render(imageIndex,*commandBuffer);

        VkImageSubresourceRange ImageSubresourceRange{};
            ImageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ImageSubresourceRange.baseMipLevel = 0;
            ImageSubresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            ImageSubresourceRange.baseArrayLayer = 0;
            ImageSubresourceRange.layerCount = 1;
        VkClearColorValue clearColorValue{};
            clearColorValue.uint32[0] = 0;
            clearColorValue.uint32[1] = 0;
            clearColorValue.uint32[2] = 0;
            clearColorValue.uint32[3] = 0;

        std::vector<VkImage> blitImages(blitAttachmentCount);
        blitImages[0] = combineBloomAttachment.image[imageIndex];
        for(size_t i=1;i<blitAttachmentCount;i++){
            blitImages[i] = blitAttachments[i-1].image[imageIndex];
        }
        VkImage blitBufferImage = blitAttachment.image[imageIndex];
        uint32_t width = extent.width;
        uint32_t height = extent.height;

        for(uint32_t k=0;k<blitAttachmentCount;k++){
            transitionImageLayout(commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_REMAINING_MIP_LEVELS);
            vkCmdClearColorImage(*commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ,&clearColorValue,1,&ImageSubresourceRange);
            blitDown(commandBuffer,blitImages[k],blitBufferImage,width,height,blitFactor);
            transitionImageLayout(commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);
            Filter.render(imageIndex,*commandBuffer,k);
        }
        for(uint32_t k=0;k<blitAttachmentCount;k++)
            transitionImageLayout(commandBuffer,blitAttachments[k].image[imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);

    PostProcessing.render(imageIndex,*commandBuffer);

    if (vkEndCommandBuffer(*commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
}

VkCommandBuffer* deferredGraphicsInterface::getCommandBuffers(uint32_t& commandBuffersCount, uint32_t imageIndex)
{
    commandBufferSet.clear();

    DeferredGraphics.getSpotLightCommandbuffers(commandBufferSet,imageIndex);
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
        DeferredGraphics.updateSpotLightCmd(imageIndex);
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
        DeferredGraphics.updateSpotLightUbo(imageIndex);
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

void                                deferredGraphicsInterface::bindLightSource(spotLight* lightSource)
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
    if(lightSource->isShadowEnable()){
        lightSource->createShadowCommandBuffers();
    }

    lightSource->createDescriptorPool(devicesInfo[0].device, imageCount);
    lightSource->createDescriptorSets(devicesInfo[0].device, imageCount);
    lightSource->updateDescriptorSets(devicesInfo[0].device, imageCount,DeferredGraphics.getEmptyTexture());

    DeferredGraphics.addSpotLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].addSpotLightSource(lightSource);
}
void                                deferredGraphicsInterface::removeLightSource(spotLight* lightSource)
{
    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(devicesInfo[0].device);
    }
    lightSource->destroyUniformBuffers(devicesInfo[0].device);
    lightSource->destroy(devicesInfo[0].device);

    DeferredGraphics.removeSpotLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].removeSpotLightSource(lightSource);
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
