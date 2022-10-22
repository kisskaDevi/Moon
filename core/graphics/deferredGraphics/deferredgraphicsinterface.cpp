#include "deferredgraphicsinterface.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/transformational/object.h"

#include <iostream>

deferredGraphicsInterface::deferredGraphicsInterface(const std::string& ExternalPath)
{
    this->ExternalPath = ExternalPath;
}

deferredGraphicsInterface::~deferredGraphicsInterface()
{}

void deferredGraphicsInterface::destroyEmptyTextures()
{
    DeferredGraphics.destroyEmptyTexture();
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].destroyEmptyTexture();
}

void deferredGraphicsInterface::destroyGraphics()
{
    DeferredGraphics.destroy();
    Filter.destroy();
    SSAO.destroy();
    SSLR.destroy();
    PostProcessing.destroy();
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].destroy();
}

VkSwapchainKHR& deferredGraphicsInterface::getSwapChain()
{
    return PostProcessing.SwapChain();
}

void deferredGraphicsInterface::createGraphics(uint32_t& imageCount, GLFWwindow* window, VkSurfaceKHR surface, VkExtent2D extent, VkSampleCountFlagBits MSAASamples, uint32_t devicesInfoCount, deviceInfo* devicesInfo)
{
    this->devicesInfo.resize(devicesInfoCount);
    for(uint32_t i=0;i<devicesInfoCount;i++){
        this->devicesInfo[i] = devicesInfo[i];
    }

    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(*devicesInfo[0].physicalDevice,surface);                   //здест происходит запрос поддерживаемы режимов и форматов которые в следующий строчках передаются в соответствующие переменные через фукцнии
    imageCount = swapChainSupport.capabilities.minImageCount + 1;                                                               //запрос на поддержк уминимального количества числа изображений, число изображений равное 2 означает что один буфер передний, а второй задний
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)            //в первом условии мы проверяем доступно ли нам вообще какое-то количество изображений и проверяем не совпадает ли максимальное число изображений с минимальным
        imageCount = swapChainSupport.capabilities.maxImageCount;

    this->imageCount = imageCount;

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);

    if(extent.height==0&&extent.width==0)
        extent = chooseSwapExtent(window, swapChainSupport.capabilities);

    if(MSAASamples != VK_SAMPLE_COUNT_1_BIT){
        VkSampleCountFlagBits maxMSAASamples = getMaxUsableSampleCount(*devicesInfo[0].physicalDevice);
        if(MSAASamples>maxMSAASamples)  MSAASamples = maxMSAASamples;
    }

    DeferredGraphics.setExternalPath(ExternalPath);
    PostProcessing.setExternalPath(ExternalPath);
    Filter.setExternalPath(ExternalPath);
    SSLR.setExternalPath(ExternalPath);
    SSAO.setExternalPath(ExternalPath);

    QueueFamilyIndices indices{*devicesInfo[0].graphicsFamily,*devicesInfo[0].presentFamily};
    DeferredGraphics.setDeviceProp(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool);
    PostProcessing.setDeviceProp(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool,&indices,&surface);
    Filter.setDeviceProp(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool);
    SSLR.setDeviceProp(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool);
    SSAO.setDeviceProp(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool);

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

    PostProcessing.createAttachments(window, swapChainSupport);
    PostProcessing.createRenderPass();
    PostProcessing.createFramebuffers();
    PostProcessing.createPipelines();
    PostProcessing.createDescriptorPool();
    PostProcessing.createDescriptorSets(DeferredGraphics.getDeferredAttachments());

    Filter.setBlitAttachments(&PostProcessing.getBlitAttachment());
    Filter.setAttachments(PostProcessing.getBlitAttachments().size(),PostProcessing.getBlitAttachments().data());
    Filter.createRenderPass();
    Filter.createFramebuffers();
    Filter.createPipelines();
    Filter.createDescriptorPool();
    Filter.createDescriptorSets();
    Filter.updateSecondDescriptorSets();

    SSLR.setSSLRAttachments(&PostProcessing.getSSLRAttachment());
    SSLR.createRenderPass();
    SSLR.createFramebuffers();
    SSLR.createPipelines();
    SSLR.createDescriptorPool();
    SSLR.createDescriptorSets();
    SSLR.updateSecondDescriptorSets(DeferredGraphics.getDeferredAttachments(),DeferredGraphics.getSceneBuffer().data());

    SSAO.setSSAOAttachments(&PostProcessing.getSSAOAttachment());
    SSAO.createRenderPass();
    SSAO.createFramebuffers();
    SSAO.createPipelines();
    SSAO.createDescriptorPool();
    SSAO.createDescriptorSets();
    SSAO.updateSecondDescriptorSets(DeferredGraphics.getDeferredAttachments(),DeferredGraphics.getSceneBuffer().data());

//    TransparentLayers.resize(TransparentLayersCount);
//    for(uint32_t i=0;i<TransparentLayersCount;i++){
//        TransparentLayers[i].setDeviceProp(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool);
//        TransparentLayers[i].setImageProp(&info);
//        TransparentLayers[i].setTransparencyPass(true);
//        TransparentLayers[i].createAttachments();
//        TransparentLayers[i].createRenderPass();
//        TransparentLayers[i].createFramebuffers();
//        TransparentLayers[i].createPipelines();
//        TransparentLayers[i].createStorageBuffers(imageCount);
//        TransparentLayers[i].createBaseDescriptorPool();
//        TransparentLayers[i].createBaseDescriptorSets();
//        TransparentLayers[i].createSkyboxDescriptorPool();
//        TransparentLayers[i].createSkyboxDescriptorSets();
//        TransparentLayers[i].createSpotLightingDescriptorPool();
//        TransparentLayers[i].createSpotLightingDescriptorSets();
//    }
}

void deferredGraphicsInterface::updateDescriptorSets()
{
    DeferredGraphics.updateBaseDescriptorSets(nullptr);
    DeferredGraphics.updateSkyboxDescriptorSets();
    DeferredGraphics.updateSpotLightingDescriptorSets();

//    for(uint32_t i=0;i<TransparentLayersCount;i++){
//        TransparentLayers[i].updateBaseDescriptorSets(nullptr);
//        TransparentLayers[i].updateSkyboxDescriptorSets();
//        TransparentLayers[i].updateSpotLightingDescriptorSets();
//    }
}

void deferredGraphicsInterface::updateCommandBuffers(uint32_t imageCount, VkCommandBuffer* commandBuffers)
{
    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        updateCommandBuffer(imageIndex, &commandBuffers[imageIndex]);

    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        DeferredGraphics.updateSpotLightCmd(imageIndex);

//    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++){
//        for(uint32_t i=0;i<TransparentLayersCount;i++){
//            TransparentLayers[i].updateSpotLightCmd(imageIndex);
//        }
//    }
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

        std::vector<VkImage> blitImages(PostProcessing.getBlitAttachments().size());
        blitImages[0] = DeferredGraphics.getDeferredAttachments().bloom->image[imageIndex];
        for(size_t i=1;i<PostProcessing.getBlitAttachments().size();i++){
            blitImages[i] = PostProcessing.getBlitAttachments()[i-1].image[imageIndex];
        }
        VkImage blitBufferImage = PostProcessing.getBlitAttachment().image[imageIndex];
        uint32_t width = PostProcessing.SwapChainImageExtent().width;
        uint32_t height = PostProcessing.SwapChainImageExtent().height;
        float blitFactor = PostProcessing.getBlitFactor();

        for(uint32_t k=0;k<PostProcessing.getBlitAttachments().size();k++){
            transitionImageLayout(commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_REMAINING_MIP_LEVELS);
            vkCmdClearColorImage(*commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ,&clearColorValue,1,&ImageSubresourceRange);
            blitDown(commandBuffer,blitImages[k],blitBufferImage,width,height,blitFactor);
            transitionImageLayout(commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);
            Filter.render(imageIndex,*commandBuffer,k);
        }
        for(uint32_t k=0;k<PostProcessing.getBlitAttachments().size();k++)
            transitionImageLayout(commandBuffer,PostProcessing.getBlitAttachments()[k].image[imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);

    PostProcessing.render(imageIndex,*commandBuffer);

    if (vkEndCommandBuffer(*commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
}

void deferredGraphicsInterface::fillCommandbufferSet(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex)
{
    DeferredGraphics.getSpotLightCommandbuffers(&commandbufferSet,imageIndex);
}

void deferredGraphicsInterface::updateCmd(uint32_t imageIndex, VkCommandBuffer* commandBuffers)
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

void deferredGraphicsInterface::updateUbo(uint32_t imageIndex)
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

void deferredGraphicsInterface::updateUniformBuffer(uint32_t imageIndex)
{
    DeferredGraphics.updateUniformBuffer(imageIndex);
    DeferredGraphics.updateSkyboxUniformBuffer(imageIndex);
    DeferredGraphics.updateObjectUniformBuffer(imageIndex);
}

void                                deferredGraphicsInterface::resetCmdLight(){lightsCmd.enable = true; lightsCmd.frames = 0;}
void                                deferredGraphicsInterface::resetCmdWorld(){worldCmd.enable = true; worldCmd.frames = 0;}
void                                deferredGraphicsInterface::resetUboLight(){lightsUbo.enable = true; lightsUbo.frames = 0;}
void                                deferredGraphicsInterface::resetUboWorld(){worldUbo.enable = true; worldUbo.frames = 0;}
void                                deferredGraphicsInterface::setExternalPath(const std::string &path){ExternalPath = path;}

void                                deferredGraphicsInterface::setEmptyTexture(std::string ZERO_TEXTURE){
    DeferredGraphics.setEmptyTexture(ZERO_TEXTURE);

//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].setEmptyTexture(ZERO_TEXTURE);
}

void                                deferredGraphicsInterface::setCameraObject(camera* cameraObject){
    DeferredGraphics.setCameraObject(cameraObject);

//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].setCameraObject(cameraObject);
}

void                                deferredGraphicsInterface::createModel(gltfModel *pModel){
    pModel->loadFromFile(devicesInfo[0].physicalDevice,devicesInfo[0].device,devicesInfo[0].queue,devicesInfo[0].commandPool,1.0f);
    pModel->createDescriptorPool(devicesInfo[0].device);
    pModel->createDescriptorSet(devicesInfo[0].device,DeferredGraphics.getEmptyTexture());
}

void                                deferredGraphicsInterface::destroyModel(gltfModel* pModel){
    pModel->destroy(devicesInfo[0].device);
}

void                                deferredGraphicsInterface::addLightSource(spotLight* lightSource)
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
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].addSpotLightSource(lightSource);
}
void                                deferredGraphicsInterface::removeLightSource(spotLight* lightSource)
{
    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(devicesInfo[0].device);
    }
    lightSource->destroyUniformBuffers(devicesInfo[0].device);
    lightSource->destroy(devicesInfo[0].device);

    DeferredGraphics.removeSpotLightSource(lightSource);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].removeSpotLightSource(lightSource);
}

void                                deferredGraphicsInterface::bindBaseObject(object* newObject)
{
    newObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    newObject->createDescriptorPool(devicesInfo[0].device,imageCount);
    newObject->createDescriptorSet(devicesInfo[0].device,imageCount);
    DeferredGraphics.bindBaseObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindBaseObject(newObject);
}
void                                deferredGraphicsInterface::bindBloomObject(object* newObject)
{
    newObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    newObject->createDescriptorPool(devicesInfo[0].device,imageCount);
    newObject->createDescriptorSet(devicesInfo[0].device,imageCount);
    DeferredGraphics.bindBloomObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindBloomObject(newObject);
}
void                                deferredGraphicsInterface::bindOneColorObject(object* newObject)
{
    newObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    newObject->createDescriptorPool(devicesInfo[0].device,imageCount);
    newObject->createDescriptorSet(devicesInfo[0].device,imageCount);
    DeferredGraphics.bindOneColorObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindOneColorObject(newObject);
}
void                                deferredGraphicsInterface::bindStencilObject(object* newObject, float lineWidth, glm::vec4 lineColor)
{
    newObject->setStencilEnable(false);
    newObject->setStencilWidth(lineWidth);
    newObject->setStencilColor(lineColor);
    newObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    newObject->createDescriptorPool(devicesInfo[0].device,imageCount);
    newObject->createDescriptorSet(devicesInfo[0].device,imageCount);
    DeferredGraphics.bindStencilObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindStencilObject(newObject);
}
void                                deferredGraphicsInterface::bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH)
{
    newObject->createUniformBuffers(devicesInfo[0].physicalDevice,devicesInfo[0].device,imageCount);
    DeferredGraphics.bindSkyBoxObject(newObject,TEXTURE_PATH);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindSkyBoxObject(newObject,TEXTURE_PATH);
}

bool                                deferredGraphicsInterface::removeBaseObject(object* object)
{
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);
    return DeferredGraphics.removeBaseObject(object);
}
bool                                deferredGraphicsInterface::removeBloomObject(object* object)
{
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);
    return DeferredGraphics.removeBloomObject(object);
}
bool                                deferredGraphicsInterface::removeOneColorObject(object* object)
{
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);
    return DeferredGraphics.removeOneColorObject(object);
}
bool                                deferredGraphicsInterface::removeStencilObject(object* object)
{
    object->destroy(devicesInfo[0].device);
    object->destroyUniformBuffers(devicesInfo[0].device);
    return DeferredGraphics.removeStencilObject(object);
}
bool                                deferredGraphicsInterface::removeSkyBoxObject(object* object)
{
    object->destroyUniformBuffers(devicesInfo[0].device);
    return DeferredGraphics.removeSkyBoxObject(object);
}

void                                deferredGraphicsInterface::removeBinds(){
    DeferredGraphics.removeBinds();
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].removeBinds();
}

void                                deferredGraphicsInterface::setMinAmbientFactor(const float& minAmbientFactor){
    DeferredGraphics.setMinAmbientFactor(minAmbientFactor);
    //for(uint32_t i=0;i<TransparentLayersCount;i++)
    //    TransparentLayers[i].setMinAmbientFactor(minAmbientFactor);
}

void                                deferredGraphicsInterface::updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition){
    DeferredGraphics.updateStorageBuffer(currentImage,mousePosition);
}
uint32_t                            deferredGraphicsInterface::readStorageBuffer(uint32_t currentImage){
    return DeferredGraphics.readStorageBuffer(currentImage);
}
