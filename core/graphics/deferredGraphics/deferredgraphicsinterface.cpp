#include "deferredgraphicsinterface.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/lightInterface.h"
#include "core/transformational/object.h"
#include "core/transformational/camera.h"

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
    Shadow.setExternalPath(ExternalPath);
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
    Blur.freeCommandBuffer(commandPool);
    Filter.freeCommandBuffer(commandPool);
    LayersCombiner.freeCommandBuffer(commandPool);
    Shadow.freeCommandBuffer(commandPool);
    Skybox.freeCommandBuffer(commandPool);
    SSAO.freeCommandBuffer(commandPool);
    SSLR.freeCommandBuffer(commandPool);
    DeferredGraphics.freeCommandBuffer(commandPool);
    for(auto& layer: TransparentLayers)
        layer.freeCommandBuffer(commandPool);

    for(auto& semaphore: semaphores){
        for(auto& semaphoresPerFrame: semaphore){
            vkDestroySemaphore(devices[0].getLogical(), semaphoresPerFrame, nullptr);
        }
        semaphore.resize(0);
    }
    semaphores.resize(0);
}

void deferredGraphicsInterface::destroyEmptyTextures()
{
    emptyTexture->destroy(&device.getLogical());
    emptyTexture = nullptr;
}

void deferredGraphicsInterface::destroyGraphics()
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
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].destroy();

    blurAttachment.deleteAttachment(&device.getLogical());
    blurAttachment.deleteSampler(&device.getLogical());

    for(auto& attachment: blitAttachments){
        attachment.deleteAttachment(&device.getLogical());
        attachment.deleteSampler(&device.getLogical());
    }

    ssaoAttachment.deleteAttachment(&device.getLogical());
    ssaoAttachment.deleteSampler(&device.getLogical());

    sslrAttachment.deleteAttachment(&device.getLogical());
    sslrAttachment.deleteSampler(&device.getLogical());

    skyboxAttachment.deleteAttachment(&device.getLogical());
    skyboxAttachment.deleteSampler(&device.getLogical());

    for(auto& attachment: layersCombinedAttachment){
        attachment.deleteAttachment(&device.getLogical());
        attachment.deleteSampler(&device.getLogical());
    }

    deferredAttachments.deleteAttachment(&device.getLogical());
    deferredAttachments.deleteSampler(&device.getLogical());
    for(auto& attachment: transparentLayersAttachments){
        attachment.deleteAttachment(&device.getLogical());
        attachment.deleteSampler(&device.getLogical());
    }

    for (size_t i = 0; i < storageBuffers.size(); i++){
        if(storageBuffers[i])       vkDestroyBuffer(device.getLogical(), storageBuffers[i], nullptr);
        if(storageBuffersMemory[i]) vkFreeMemory(device.getLogical(), storageBuffersMemory[i], nullptr);
    }
    storageBuffers.resize(0);

    if(swapChain) {vkDestroySwapchainKHR(device.getLogical(), swapChain, nullptr); swapChain = VK_NULL_HANDLE;}
}

void deferredGraphicsInterface::destroyCommandPool()
{
    if(commandPool) {vkDestroyCommandPool(device.getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;}
}

void deferredGraphicsInterface::setDevices(uint32_t devicesCount, physicalDevice* devices)
{
    this->devices.resize(devicesCount);
    for(uint32_t i=0;i<devicesCount;i++){
        this->devices[i] = devices[i];
    }

    device = this->devices[0];

    DeferredGraphics.setDeviceProp(&device.instance, &device.getLogical());
    PostProcessing.setDeviceProp(&device.instance, &device.getLogical());
    Filter.setDeviceProp(&device.instance, &device.getLogical());
    SSLR.setDeviceProp(&device.instance, &device.getLogical());
    SSAO.setDeviceProp(&device.instance, &device.getLogical());
    Skybox.setDeviceProp(&device.instance, &device.getLogical());
    Shadow.setDeviceProp(&device.instance, &device.getLogical());
    LayersCombiner.setDeviceProp(&device.instance, &device.getLogical());
    Blur.setDeviceProp(&device.instance, &device.getLogical());

    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].setDeviceProp(&device.instance, &device.getLogical());
}

void deferredGraphicsInterface::setSupportImageCount(VkSurfaceKHR* surface)
{
    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(device.instance, *surface);
    imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    updateCommandBufferFlags.resize(imageCount,true);
}

void deferredGraphicsInterface::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);
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
    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(device.instance,*surface);
    VkSurfaceFormatKHR      surfaceFormat = SwapChain::queryingSurfaceFormat(swapChainSupport.formats);

    if(extent.height==0&&extent.width==0){
        extent = SwapChain::queryingExtent(window, swapChainSupport.capabilities);
    }

    if(MSAASamples != VK_SAMPLE_COUNT_1_BIT){
        VkSampleCountFlagBits maxMSAASamples = PhysicalDevice::queryingSampleCount(device.instance);
        if(MSAASamples>maxMSAASamples)  MSAASamples = maxMSAASamples;
    }

    imageInfo shadowsInfo{imageCount,VK_FORMAT_D32_SFLOAT,VkExtent2D{1024,1024},MSAASamples};
    Shadow.setImageProp(&shadowsInfo);

    imageInfo info{imageCount, surfaceFormat.format, extent, MSAASamples};
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

    Shadow.createRenderPass();
    Shadow.createPipelines();

    std::vector<uint32_t> queueIndices = {0};
    PostProcessing.setLayersAttachment(enableTransparentLayers ? &layersCombinedAttachment[0] : &deferredAttachments.image);
    PostProcessing.createSwapChain(&swapChain, window, swapChainSupport, surface, static_cast<uint32_t>(queueIndices.size()), queueIndices.data());
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

    commandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(imageCount);
    vkAllocateCommandBuffers(devices[0].getLogical(), &allocInfo, commandBuffers.data());

    updateCmdFlags();

    semaphores.resize(1);

    for (auto& semaphore: semaphores){
        semaphore.resize(imageCount);
        for (size_t imageIndex = 0; imageIndex < imageCount; imageIndex++){
            VkSemaphoreCreateInfo semaphoreInfo{};
                semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            vkCreateSemaphore(devices[0].getLogical(), &semaphoreInfo, nullptr, &semaphore[imageIndex]);
        }
    }
}

void deferredGraphicsInterface::updateDescriptorSets()
{
    DeferredGraphics.updateDescriptorSets(nullptr, storageBuffers.data(), sizeof(StorageBufferObject), cameraObject);

    if(enableTransparentLayers){
        TransparentLayers[0].updateDescriptorSets(nullptr, storageBuffers.data(), sizeof(StorageBufferObject), cameraObject);
        for(uint32_t i=1;i<TransparentLayers.size();i++){
            TransparentLayers[i].updateDescriptorSets(&transparentLayersAttachments[i-1].depth, storageBuffers.data(), sizeof(StorageBufferObject), cameraObject);
        }
    }
}

void deferredGraphicsInterface::updateCommandBuffers()
{
    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        updateCommandBuffer(imageIndex);
}

VkSemaphore deferredGraphicsInterface::sibmit(VkSemaphore externalSemaphore, VkFence& externalFence, uint32_t imageIndex)
{
    std::vector<stage> stages;
    stages.push_back(stage( {commandBuffers[imageIndex],
                            Shadow.getCommandBuffer(imageIndex), Skybox.getCommandBuffer(imageIndex),
                            DeferredGraphics.getCommandBuffer(imageIndex),
                            TransparentLayers[0].getCommandBuffer(imageIndex), TransparentLayers[1].getCommandBuffer(imageIndex),
                            LayersCombiner.getCommandBuffer(imageIndex),
                            Filter.getCommandBuffer(imageIndex),
                            Blur.getCommandBuffer(imageIndex),
                            SSLR.getCommandBuffer(imageIndex),
                            SSAO.getCommandBuffer(imageIndex),
                            PostProcessing.getCommandBuffer(imageIndex)},
                            {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
                            {externalSemaphore},
                            {semaphores[0][imageIndex]},
                            devices[0].getQueue(0,0),
                            externalFence));

    for(auto& stage: stages){
        stage.submit();
    }

    return semaphores[0][imageIndex];
}

void deferredGraphicsInterface::updateCommandBuffer(uint32_t imageIndex)
{
    if(updateCommandBufferFlags[imageIndex]){

        Shadow.updateCommandBuffer(imageIndex);
        Skybox.updateCommandBuffer(imageIndex);
        DeferredGraphics.updateCommandBuffer(imageIndex);
        if(enableTransparentLayers)
            for(auto& layer: TransparentLayers)
               layer.updateCommandBuffer(imageIndex);
        Blur.updateCommandBuffer(imageIndex);
        SSLR.updateCommandBuffer(imageIndex);
        SSAO.updateCommandBuffer(imageIndex);
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
        Filter.updateCommandBuffer(imageIndex);
        PostProcessing.updateCommandBuffer(imageIndex);

        updateCommandBufferFlags[imageIndex] = false;
    }
}

void deferredGraphicsInterface::updateBuffers(uint32_t imageIndex)
{
    vkResetCommandBuffer(commandBuffers[imageIndex],0);

     VkCommandBufferBeginInfo beginInfo{};
         beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
         beginInfo.flags = 0;
         beginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo);

    cameraObject->updateUniformBuffer(commandBuffers[imageIndex], imageIndex);
    Skybox.updateObjectUniformBuffer(commandBuffers[imageIndex], imageIndex);
    DeferredGraphics.updateObjectUniformBuffer(commandBuffers[imageIndex], imageIndex);
    DeferredGraphics.updateLightSourcesUniformBuffer(commandBuffers[imageIndex], imageIndex);

    vkEndCommandBuffer(commandBuffers[imageIndex]);
}

void deferredGraphicsInterface::createStorageBuffers(uint32_t imageCount)
{
    storageBuffers.resize(imageCount);
    storageBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        Buffer::create( device.instance,
                        device.getLogical(),
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
    vkMapMemory(device.getLogical(), storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(data, &StorageUBO, sizeof(StorageUBO));
    vkUnmapMemory(device.getLogical(), storageBuffersMemory[currentImage]);
}

uint32_t deferredGraphicsInterface::readStorageBuffer(uint32_t currentImage){
    void* data;

    StorageBufferObject StorageUBO{};
    vkMapMemory(device.getLogical(), storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(&StorageUBO, data, sizeof(StorageUBO));
    vkUnmapMemory(device.getLogical(), storageBuffersMemory[currentImage]);

    return StorageUBO.number;
}

uint32_t deferredGraphicsInterface::getImageCount()       {   return imageCount;}
VkSwapchainKHR& deferredGraphicsInterface::getSwapChain() {   return swapChain;}

void deferredGraphicsInterface::setExtent(VkExtent2D extent)             {   this->extent = extent;}
void deferredGraphicsInterface::setExternalPath(const std::string &path) {   ExternalPath = path;}
void deferredGraphicsInterface::setEmptyTexture(std::string ZERO_TEXTURE){
    this->emptyTexture = new texture(ZERO_TEXTURE);

    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    emptyTexture->createTextureImage(device.instance, device.getLogical(), commandBuffer);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
    emptyTexture->destroyStagingBuffer(&device.getLogical());

    emptyTexture->createTextureImageView(&device.getLogical());
    emptyTexture->createTextureSampler(&device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});

    DeferredGraphics.setEmptyTexture(emptyTexture);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].setEmptyTexture(emptyTexture);

    Blur.setEmptyTexture(emptyTexture);
    Filter.setEmptyTexture(emptyTexture);
    LayersCombiner.setEmptyTexture(emptyTexture);
    Skybox.setEmptyTexture(emptyTexture);
    SSAO.setEmptyTexture(emptyTexture);
    SSLR.setEmptyTexture(emptyTexture);
    Shadow.setEmptyTexture(emptyTexture);
    PostProcessing.setEmptyTexture(emptyTexture);
}

void deferredGraphicsInterface::createModel(gltfModel *pModel){
    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    pModel->loadFromFile(device.instance, device.getLogical(), commandBuffer);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0), commandPool, &commandBuffer);
    pModel->destroyStagingBuffer(device.getLogical());
    pModel->createDescriptorPool(&device.getLogical());
    pModel->createDescriptorSet(&device.getLogical(), emptyTexture);
}

void deferredGraphicsInterface::destroyModel(gltfModel* pModel){
    pModel->destroy(&device.getLogical());
}

void deferredGraphicsInterface::bindCameraObject(camera* cameraObject){
    this->cameraObject = cameraObject;
    cameraObject->createUniformBuffers(&device.instance,&device.getLogical(),imageCount);
}

void deferredGraphicsInterface::removeCameraObject(camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject->destroy(&device.getLogical());
        this->cameraObject = nullptr;
    }
}

void deferredGraphicsInterface::bindLightSource(light* lightSource){
    if(lightSource->getTexture()){
        VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
        lightSource->getTexture()->createTextureImage(device.instance, device.getLogical(), commandBuffer);
        SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
        lightSource->getTexture()->destroyStagingBuffer(&device.getLogical());
        lightSource->getTexture()->createTextureImageView(&device.getLogical());
        lightSource->getTexture()->createTextureSampler(&device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    }
    if(lightSource->isShadowEnable()){
        Shadow.addLightSource(lightSource);
        Shadow.createAttachments(1,lightSource->getAttachments());
        Shadow.setAttachments(1,lightSource->getAttachments());
        Shadow.createFramebuffers(lightSource);
    }

    lightSource->createUniformBuffers(&device.instance,&device.getLogical(),imageCount);

    lightSource->createDescriptorPool(&device.getLogical(), imageCount);
    lightSource->createDescriptorSets(&device.getLogical(), imageCount);
    lightSource->updateDescriptorSets(&device.getLogical(), imageCount, emptyTexture);

    DeferredGraphics.addLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].addLightSource(lightSource);

    updateCmdFlags();
}

void deferredGraphicsInterface::removeLightSource(light* lightSource){
    if(lightSource->getAttachments()){
        lightSource->getAttachments()->deleteAttachment(&device.getLogical());
        lightSource->getAttachments()->deleteSampler(&device.getLogical());
    }
    lightSource->destroy(&device.getLogical());

    DeferredGraphics.removeLightSource(lightSource);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].removeLightSource(lightSource);

    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(&device.getLogical());
    }
    Shadow.removeLightSource(lightSource);

    updateCmdFlags();
}

void deferredGraphicsInterface::bindBaseObject(object* object){
    object->createUniformBuffers(device.instance,device.getLogical(),imageCount);
    object->createDescriptorPool(device.getLogical(),imageCount);
    object->createDescriptorSet(device.getLogical(),imageCount);

    DeferredGraphics.bindBaseObject(object);
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        TransparentLayers[i].bindBaseObject(object);

    Shadow.bindBaseObject(object);

    updateCmdFlags();
}

bool deferredGraphicsInterface::removeObject(object* object){
    object->destroy(device.getLogical());

    bool res = true;
    for(uint32_t i=0;i<TransparentLayers.size();i++)
        res = res&&(TransparentLayers[i].removeBaseObject(object));

    Shadow.removeBaseObject(object);

    updateCmdFlags();

    return res&&(DeferredGraphics.removeBaseObject(object));
}

void deferredGraphicsInterface::bindSkyBoxObject(skyboxObject* object){
    object->createUniformBuffers(device.instance,device.getLogical(),imageCount);
    if(object->getTexture()){
        VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
        object->getTexture()->createTextureImage(device.instance, device.getLogical(), commandBuffer);
        SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
        object->getTexture()->createTextureImageView(&device.getLogical());
        object->getTexture()->createTextureSampler(&device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        object->getTexture()->destroyStagingBuffer(&device.getLogical());
    }

    object->createDescriptorPool(device.getLogical(),imageCount);
    object->createDescriptorSet(device.getLogical(),imageCount);

    Skybox.bindObject(object);
}

bool deferredGraphicsInterface::removeSkyBoxObject(skyboxObject* object){
    object->destroy(device.getLogical());
    if(object->getTexture()){
        object->getTexture()->destroy(&device.getLogical());
    }

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
