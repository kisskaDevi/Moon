#include "deferredGraphics.h"
#include "operations.h"
#include "texture.h"
#include "node.h"
#include "model.h"
#include "light.h"
#include "object.h"
#include "camera.h"
#include "swapChain.h"

#include <cstring>

deferredGraphics::deferredGraphics(const std::filesystem::path& shadersPath, VkExtent2D extent, VkOffset2D offset, VkSampleCountFlagBits MSAASamples):
    shadersPath(shadersPath), extent(extent), offset(offset), MSAASamples(MSAASamples)
{
    DeferredGraphics.setShadersPath(shadersPath);
    LayersCombiner.setShadersPath(shadersPath);
    PostProcessing.setShadersPath(shadersPath);
    Link.setShadersPath(shadersPath);

    Filter.setShadersPath(shadersPath);
    SSLR.setShadersPath(shadersPath);
    SSAO.setShadersPath(shadersPath);
    Skybox.setShadersPath(shadersPath);
    Blur.setShadersPath(shadersPath);
    Scattering.setShadersPath(shadersPath);
    Shadow.setShadersPath(shadersPath);
    BoundingBox.setShadersPath(shadersPath);

    TransparentLayers.resize(TransparentLayersCount);
    for(auto& layer: TransparentLayers){
        layer.setShadersPath(shadersPath);
    }
}

void deferredGraphics::destroyEmptyTextures(){
    for(auto& [_,texture] : emptyTextures){
        if(texture){
            texture->destroy(device.getLogical());
            texture = nullptr;
        }
    }
}

void deferredGraphics::freeCommandBuffers(){
    CHECKERROR(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::freeCommandBuffers ] commandPool is VK_NULL_HANDLE"));
    CHECKERROR(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::freeCommandBuffers ] VkDevice is VK_NULL_HANDLE"));

    Blur.freeCommandBuffer(commandPool);
    Filter.freeCommandBuffer(commandPool);
    LayersCombiner.freeCommandBuffer(commandPool);
    Shadow.freeCommandBuffer(commandPool);
    Skybox.freeCommandBuffer(commandPool);
    SSAO.freeCommandBuffer(commandPool);
    SSLR.freeCommandBuffer(commandPool);
    Scattering.freeCommandBuffer(commandPool);
    BoundingBox.freeCommandBuffer(commandPool);
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

void deferredGraphics::destroyGraphics(){
    freeCommandBuffers();
    destroyCommandPool();
    destroyEmptyTextures();

    DeferredGraphics.destroy();
    Filter.destroy();
    SSAO.destroy();
    SSLR.destroy();
    Skybox.destroy();
    Shadow.destroy();
    PostProcessing.destroy();
    LayersCombiner.destroy();
    Blur.destroy();
    Scattering.destroy();
    BoundingBox.destroy();
    for(auto& layer: TransparentLayers){
        layer.destroy();
    }
    Link.destroy();

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

    scatteringAttachment.deleteAttachment(device.getLogical());
    scatteringAttachment.deleteSampler(device.getLogical());

    skyboxAttachment.deleteAttachment(device.getLogical());
    skyboxAttachment.deleteSampler(device.getLogical());

    combinedAttachment.deleteAttachment(device.getLogical());
    combinedAttachment.deleteSampler(device.getLogical());

    deferredAttachments.deleteAttachment(device.getLogical());
    deferredAttachments.deleteSampler(device.getLogical());
    for(auto& attachment: transparentLayersAttachments){
        attachment.deleteAttachment(device.getLogical());
        attachment.deleteSampler(device.getLogical());
    }

    finalAttachment.deleteAttachment(device.getLogical());
    finalAttachment.deleteSampler(device.getLogical());

    boundingBoxAttachment.deleteAttachment(device.getLogical());
    boundingBoxAttachment.deleteSampler(device.getLogical());

    for (auto& buffer: storageBuffersHost){
        buffer.destroy(device.getLogical());
    }
    storageBuffersHost.clear();
}

void deferredGraphics::destroyCommandPool(){
    if(commandPool){
        vkDestroyCommandPool(device.getLogical(), commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
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
    Scattering.setDeviceProp(device.instance, device.getLogical());
    BoundingBox.setDeviceProp(device.instance, device.getLogical());
    for(auto& layer: TransparentLayers){
        layer.setDeviceProp(device.instance, device.getLogical());
    }
    Link.setDeviceProp(device.getLogical());
}

void deferredGraphics::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);
}

void deferredGraphics::setSwapChain(swapChain* swapChainKHR)
{
    this->swapChainKHR = swapChainKHR;
    this->imageCount = swapChainKHR->getImageCount();
}

void deferredGraphics::createGraphics()
{
    createCommandPool();
    createEmptyTexture();
    createGraphicsPasses();
    createCommandBuffers();
}

namespace {
    void fastCreateFilterGraphics(workflow* workflow, uint32_t attachmentsNumber, attachments* attachments)
    {
        workflow->setAttachments(attachmentsNumber,attachments);
        workflow->createRenderPass();
        workflow->createFramebuffers();
        workflow->createPipelines();
        workflow->createDescriptorPool();
        workflow->createDescriptorSets();
    }

    void fastCreateGraphics(graphics* graphics, DeferredAttachments* attachments)
    {
        graphics->createAttachments(attachments);
        graphics->setAttachments(attachments);
        graphics->createRenderPass();
        graphics->createFramebuffers();
        graphics->createPipelines();
        graphics->createDescriptorPool();
        graphics->createDescriptorSets();
    }
}

void deferredGraphics::createGraphicsPasses(){
    CHECKERROR(commandPool == VK_NULL_HANDLE,       std::string("[ deferredGraphics::createGraphicsPasses ] VkCommandPool is VK_NULL_HANDLE"));
    CHECKERROR(device.instance == VK_NULL_HANDLE,   std::string("[ deferredGraphics::createGraphicsPasses ] VkPhysicalDevice is VK_NULL_HANDLE"));
    CHECKERROR(swapChainKHR == nullptr,             std::string("[ deferredGraphics::createGraphicsPasses ] swapChain is nullptr"));
    CHECKERROR(cameraObject == nullptr,             std::string("[ deferredGraphics::createGraphicsPasses ] camera is nullptr"));

    imageCount = imageCount == 0 ? SwapChain::queryingSupportImageCount(device.instance, swapChainKHR->getSurface()) : imageCount;

    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(device.instance, swapChainKHR->getSurface());

    frameBufferExtent = swapChainKHR->getExtent();

    imageInfo shadowsInfo{imageCount,VK_FORMAT_D32_SFLOAT,VkOffset2D{0,0},VkExtent2D{1024,1024},VkExtent2D{1024,1024},MSAASamples};
    Shadow.setImageProp(&shadowsInfo);

    imageInfo info{imageCount, SwapChain::queryingSurfaceFormat(swapChainSupport.formats).format, VkOffset2D{0,0}, extent, extent, MSAASamples};

    DeferredGraphics.setImageProp(&info);
    Blur.setImageProp(&info);
    LayersCombiner.setImageProp(&info);
    Filter.setImageProp(&info);
    SSLR.setImageProp(&info);
    SSAO.setImageProp(&info);
    Skybox.setImageProp(&info);
    Scattering.setImageProp(&info);
    BoundingBox.setImageProp(&info);
    for(auto& layer: TransparentLayers){
        layer.setImageProp(&info);
    }
    Link.setImageCount(imageCount);

    imageInfo swapChainInfo{imageCount, SwapChain::queryingSurfaceFormat(swapChainSupport.formats).format, offset, extent, frameBufferExtent, MSAASamples};

    PostProcessing.setImageProp(&swapChainInfo);

    fastCreateGraphics(&DeferredGraphics, &deferredAttachments);

    if(enableTransparentLayers){
        LayersCombiner.setTransparentLayersCount(TransparentLayersCount);
    }
    LayersCombiner.setEmptyTextureWhite(emptyTextures["white"]);
    LayersCombiner.createAttachments(combinedAttachment.size(),&combinedAttachment);
    fastCreateFilterGraphics(&LayersCombiner,combinedAttachment.size(),&combinedAttachment);
    PostProcessing.setLayersAttachment(&combinedAttachment.color);

    if(enableSkybox){
        Skybox.createAttachments(skyboxAttachment.size(),&skyboxAttachment);
        fastCreateFilterGraphics(&Skybox,skyboxAttachment.size(),&skyboxAttachment);
    }

    if(enableScattering){
        Scattering.createAttachments(1,&scatteringAttachment);
        fastCreateFilterGraphics(&Scattering,1,&scatteringAttachment);
    }

    if(enableTransparentLayers){
        transparentLayersAttachments.resize(TransparentLayersCount);
        for(uint32_t i=0;i<transparentLayersAttachments.size();i++){
            TransparentLayers[i].setTransparencyPass(true);
            fastCreateGraphics(&TransparentLayers[i], &transparentLayersAttachments[i]);
        }
    }

    if(enableBloom){
        blitAttachments.resize(blitAttachmentCount);
        Filter.createBufferAttachments();
        Filter.setBlitFactor(blitFactor);
        Filter.setSrcAttachment(&combinedAttachment.bloom);
        Filter.createAttachments(blitAttachmentCount,blitAttachments.data());
        fastCreateFilterGraphics(&Filter,blitAttachmentCount,blitAttachments.data());
        PostProcessing.setBlitAttachments(blitAttachmentCount,blitAttachments.data(),blitFactor);
    }

    if(enableBlur){
        Blur.createBufferAttachments();
        Blur.createAttachments(1,&blurAttachment);
        fastCreateFilterGraphics(&Blur,1,&blurAttachment);
        PostProcessing.setBlurAttachment(&blurAttachment);
    }

    if(enableSSAO){
        SSAO.createAttachments(1,&ssaoAttachment);
        fastCreateFilterGraphics(&SSAO,1,&ssaoAttachment);
        PostProcessing.setSSAOAttachment(&ssaoAttachment);
    }

    if(enableSSLR){
        SSLR.createAttachments(1,&sslrAttachment);
        fastCreateFilterGraphics(&SSLR,1,&sslrAttachment);
        PostProcessing.setSSLRAttachment(&sslrAttachment);
    }

    if(enableShadow){
        Shadow.createRenderPass();
        Shadow.createPipelines();
    }

    if(enableBoundingBox){
        BoundingBox.createAttachments(1,&boundingBoxAttachment);
        fastCreateFilterGraphics(&BoundingBox,1,&boundingBoxAttachment);
        PostProcessing.setBoundingBoxbAttachment(&boundingBoxAttachment);
    }

    PostProcessing.createAttachments(1,&finalAttachment);
    fastCreateFilterGraphics(&PostProcessing,1,&finalAttachment);

    Link.createDescriptorSetLayout();
    Link.createPipeline(&swapChainInfo);
    Link.createDescriptorPool();
    Link.createDescriptorSets();

    createStorageBuffers(imageCount);

    updateCommandBufferFlags.resize(imageCount, true);
}

void deferredGraphics::updateDescriptorSets(){
    CHECKERROR(cameraObject == nullptr, std::string("[ deferredGraphics::updateDescriptorSets ] camera is nullptr"));

    std::vector<VkBuffer> storageBuffers;
    for(const auto& buffer: storageBuffersHost){
        storageBuffers.push_back(buffer.instance);
    }

    DeferredGraphics.updateDescriptorSets(
        nullptr,
        storageBuffers.data(),
        sizeof(StorageBufferObject),
        cameraObject
    );
    LayersCombiner.updateDescriptorSets(
        deferredAttachments,
        enableTransparentLayers ? transparentLayersAttachments.data() : nullptr,
        enableSkybox ? &skyboxAttachment.color : nullptr,
        enableSkybox ? &skyboxAttachment.bloom : nullptr,
        enableScattering ? &scatteringAttachment : nullptr,
        cameraObject
    );

    if(enableTransparentLayers){
        for(uint32_t i=0;i<TransparentLayers.size();i++){
            TransparentLayers[i].updateDescriptorSets(
                i == 0 ? nullptr : &transparentLayersAttachments[i-1].GBuffer.depth,
                storageBuffers.data(),
                sizeof(StorageBufferObject),
                cameraObject
            );
        }
    }
    if(enableSkybox){
        Skybox.updateDescriptorSets(
            cameraObject
        );
    }
    if(enableScattering){
        Scattering.updateDescriptorSets(
            cameraObject,
            &deferredAttachments.GBuffer.depth
        );
    }
    if(enableBloom){
        Filter.updateDescriptorSets();
    }
    if(enableBlur){
        Blur.updateDescriptorSets(
            &deferredAttachments.blur
        );
    }
    if(enableSSAO){
        SSAO.updateDescriptorSets(
            cameraObject,
            &deferredAttachments.GBuffer.position,
            &deferredAttachments.GBuffer.normal,
            &deferredAttachments.image,
            &deferredAttachments.GBuffer.depth
        );
    }
    if(auto& layer = enableTransparentLayers ? transparentLayersAttachments.front() : deferredAttachments; enableSSLR){
        SSLR.updateDescriptorSets(
            cameraObject,
            &deferredAttachments.GBuffer.position,
            &deferredAttachments.GBuffer.normal,
            &deferredAttachments.image,
            &deferredAttachments.GBuffer.depth,
            &layer.GBuffer.position,
            &layer.GBuffer.normal,
            &layer.image,
            &layer.GBuffer.depth
        );
    }
    if(enableBoundingBox){
        BoundingBox.updateDescriptorSets(
            cameraObject
        );
    }

    PostProcessing.updateDescriptorSets();
    Link.updateDescriptorSets(&finalAttachment);
}

void deferredGraphics::createCommandBuffers(){
    CHECKERROR(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::createCommandBuffers ] VkCommandPool is VK_NULL_HANDLE"));

    Shadow.createCommandBuffers(commandPool);
    Skybox.createCommandBuffers(commandPool);
    DeferredGraphics.createCommandBuffers(commandPool);
    for(auto& layer: TransparentLayers){
        layer.createCommandBuffers(commandPool);
    }
    Blur.createCommandBuffers(commandPool);
    SSLR.createCommandBuffers(commandPool);
    SSAO.createCommandBuffers(commandPool);
    Scattering.createCommandBuffers(commandPool);
    LayersCombiner.createCommandBuffers(commandPool);
    Filter.createCommandBuffers(commandPool);
    BoundingBox.createCommandBuffers(commandPool);
    PostProcessing.createCommandBuffers(commandPool);

    copyCommandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(imageCount);
    vkAllocateCommandBuffers(device.getLogical(), &allocInfo, copyCommandBuffers.data());

    updateCmdFlags();

    auto getTransparentLayersCommandBuffers = [this](uint32_t imageIndex) -> std::vector<VkCommandBuffer>{
        std::vector<VkCommandBuffer> commandBuffers;
        for(auto& transparentLayer: TransparentLayers){
            commandBuffers.push_back(transparentLayer.getCommandBuffer(imageIndex));
        }
        return commandBuffers;
    };

    nodes.resize(imageCount);
    for(uint32_t imageIndex = 0; imageIndex < imageCount; imageIndex++){
        nodes[imageIndex]
         = new node({
            stage(  {   copyCommandBuffers[imageIndex]},
                    {   VK_PIPELINE_STAGE_TRANSFER_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {   Shadow.getCommandBuffer(imageIndex)},
                    {   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
                    device.getQueue(0,0)),
            stage(  {   Skybox.getCommandBuffer(imageIndex)},
                    {   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {   DeferredGraphics.getCommandBuffer(imageIndex)},
                    {   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                    device.getQueue(0,0)),
            stage(  getTransparentLayersCommandBuffers(imageIndex),
                    {   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {   Scattering.getCommandBuffer(imageIndex)},
                    {   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {   LayersCombiner.getCommandBuffer(imageIndex)},
                    {   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                    device.getQueue(0,0))
        }, new node({
            stage(  {   SSLR.getCommandBuffer(imageIndex),
                        SSAO.getCommandBuffer(imageIndex),
                        Filter.getCommandBuffer(imageIndex),
                        Blur.getCommandBuffer(imageIndex),
                        BoundingBox.getCommandBuffer(imageIndex),
                        PostProcessing.getCommandBuffer(imageIndex)
                    },
                    {   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
                    device.getQueue(0,0))
        }, nullptr))))));

        nodes[imageIndex]->createSemaphores(device.getLogical());
    }
}

std::vector<std::vector<VkSemaphore>> deferredGraphics::submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex){
    if(externalSemaphore.size()){
        nodes[imageIndex]->setExternalSemaphore(externalSemaphore);
    }
    if(externalFence.size()){
        nodes[imageIndex]->back()->setExternalFence(externalFence);
    }

    nodes[imageIndex]->submit();

    return nodes[imageIndex]->back()->getBackSemaphores();
}

linkable* deferredGraphics::getLinkable() {
    return &Link;
}

void deferredGraphics::updateCommandBuffer(uint32_t imageIndex){
    if(updateCommandBufferFlags[imageIndex]){
        Shadow.beginCommandBuffer(imageIndex);
            if(enableShadow) Shadow.updateCommandBuffer(imageIndex);
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

        Scattering.beginCommandBuffer(imageIndex);
            if(enableScattering){ Scattering.updateCommandBuffer(imageIndex);}
        Scattering.endCommandBuffer(imageIndex);

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
            LayersCombiner.updateCommandBuffer(imageIndex);
        LayersCombiner.endCommandBuffer(imageIndex);

        Filter.beginCommandBuffer(imageIndex);
            if(enableBloom){ Filter.updateCommandBuffer(imageIndex);}
        Filter.endCommandBuffer(imageIndex);

        BoundingBox.beginCommandBuffer(imageIndex);
            if(enableBoundingBox){ BoundingBox.updateCommandBuffer(imageIndex);}
        BoundingBox.endCommandBuffer(imageIndex);

        PostProcessing.beginCommandBuffer(imageIndex);
            PostProcessing.updateCommandBuffer(imageIndex);
        PostProcessing.endCommandBuffer(imageIndex);

        updateCommandBufferFlags[imageIndex] = false;
    }
}

void deferredGraphics::updateBuffers(uint32_t imageIndex){
    vkResetCommandBuffer(copyCommandBuffers[imageIndex],0);

     VkCommandBufferBeginInfo beginInfo{};
         beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
         beginInfo.flags = 0;
         beginInfo.pInheritanceInfo = nullptr;
    vkBeginCommandBuffer(copyCommandBuffers[imageIndex], &beginInfo);

    cameraObject->updateUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);
    if(enableSkybox) Skybox.updateObjectUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);
    DeferredGraphics.updateObjectUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);
    DeferredGraphics.updateLightSourcesUniformBuffer(copyCommandBuffers[imageIndex], imageIndex);

    vkEndCommandBuffer(copyCommandBuffers[imageIndex]);
}

void deferredGraphics::createStorageBuffers(uint32_t imageCount){
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
        Memory::instance().nameMemory(buffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", deferredGraphics::createStorageBuffers, storageBuffersHost " + std::to_string(&buffer - &storageBuffersHost[0]));
    }
}

void deferredGraphics::updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey){
    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = vector<float,4>(mousex,mousey,0.0f,0.0f);
        StorageUBO.number = INT_FAST32_MAX;
        StorageUBO.depth = 1.0f;
    std::memcpy(storageBuffersHost[currentImage].map, &StorageUBO, sizeof(StorageUBO));
}

uint32_t deferredGraphics::readStorageBuffer(uint32_t currentImage){
    StorageBufferObject storageBuffer{};
    std::memcpy(&storageBuffer, storageBuffersHost[currentImage].map, sizeof(StorageBufferObject));
    return storageBuffer.number;
}

void deferredGraphics::setExtentAndOffset(VkExtent2D extent, VkOffset2D offset) {
    this->offset = offset;
    this->extent = extent;
}
void deferredGraphics::setFrameBufferExtent(VkExtent2D extent){
    frameBufferExtent = extent;
}
void deferredGraphics::setShadersPath(const std::filesystem::path& path){
    shadersPath = path;
}

void deferredGraphics::createEmptyTexture()
{
    CHECKERROR(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::createEmptyTexture ] VkCommandPool is VK_NULL_HANDLE"));
    CHECKERROR(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::createEmptyTexture ] VkPhysicalDevice is VK_NULL_HANDLE"));
    CHECKERROR(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::createEmptyTexture ] VkDevice is VK_NULL_HANDLE"));

    emptyTextures["black"] = ::createEmptyTexture(device, commandPool);
    emptyTextures["white"] = ::createEmptyTexture(device, commandPool, false);

    DeferredGraphics.setEmptyTexture(emptyTextures["black"]);
    LayersCombiner.setEmptyTexture(emptyTextures["black"]);
    PostProcessing.setEmptyTexture(emptyTextures["black"]);

    if(enableTransparentLayers){
        for(auto& layer: TransparentLayers){
            layer.setEmptyTexture(emptyTextures["black"]);
        }
    }
    if(enableBlur)          Blur.setEmptyTexture(emptyTextures["black"]);
    if(enableBloom)         Filter.setEmptyTexture(emptyTextures["black"]);
    if(enableSkybox)        Skybox.setEmptyTexture(emptyTextures["black"]);
    if(enableSSAO)          SSAO.setEmptyTexture(emptyTextures["black"]);
    if(enableScattering)    Scattering.setEmptyTexture(emptyTextures["black"]);
    if(enableSSLR)          SSLR.setEmptyTexture(emptyTextures["black"]);
    if(enableShadow)        Shadow.setEmptyTexture(emptyTextures["black"]);
    if(enableBoundingBox)   BoundingBox.setEmptyTexture(emptyTextures["black"]);
}

void deferredGraphics::create(model *pModel){
    pModel->create(device, commandPool);
}

void deferredGraphics::destroy(model* pModel){
    pModel->destroy(device.getLogical());
}

void deferredGraphics::bind(camera* cameraObject){
    this->cameraObject = cameraObject;
    cameraObject->create(device, imageCount);
}

void deferredGraphics::remove(camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject->destroy(device.getLogical());
        this->cameraObject = nullptr;
    }
}

void deferredGraphics::bind(light* lightSource){
    if(lightSource->isShadowEnable() && enableShadow){
        if(lightSource->getAttachments()->instances.empty()){
            Shadow.createAttachments(1,lightSource->getAttachments());
        }
        Shadow.bindLightSource(lightSource);
        Shadow.createFramebuffers(lightSource);
    }
    lightSource->create(device, commandPool, imageCount);

    DeferredGraphics.bind(lightSource);
    for(auto& TransparentLayer: TransparentLayers){
        TransparentLayer.bind(lightSource);
    }
    Scattering.bindLightSource(lightSource);

    updateCmdFlags();
}

void deferredGraphics::remove(light* lightSource){
    if(lightSource->getAttachments()){
        lightSource->getAttachments()->deleteAttachment(device.getLogical());
        lightSource->getAttachments()->deleteSampler(device.getLogical());
    }
    lightSource->destroy(device.getLogical());

    DeferredGraphics.remove(lightSource);
    for(auto& TransparentLayer: TransparentLayers){
        TransparentLayer.remove(lightSource);
    }
    Scattering.removeLightSource(lightSource);

    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(device.getLogical());
    }
    Shadow.removeLightSource(lightSource);

    updateCmdFlags();
}

void deferredGraphics::bind(object* object){
    object->create(device, commandPool, imageCount);

    switch (object->getPipelineBitMask()) {
        case (0<<4)|0x0:
        case (1<<4)|0x0:
            Shadow.bindBaseObject(object);
            DeferredGraphics.bind(object);
            for(auto& layer: TransparentLayers){
                layer.bind(object);
            }
            BoundingBox.bindObject(object);
            break;
        case (0<<4)|0x1:
            Skybox.bindObject(object);
            break;
    }

    updateCmdFlags();
}

bool deferredGraphics::remove(object* object){
    object->destroy(device.getLogical());

    bool res = true;

    switch (object->getPipelineBitMask()) {
        case (0<<4)|0x0:
        case (1<<4)|0x0:
            res = res && Shadow.removeBaseObject(object) && DeferredGraphics.remove(object) && BoundingBox.removeObject(object);
            for(auto& layer: TransparentLayers){
                res = res && layer.remove(object);
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

void deferredGraphics::setScatteringRefraction(bool enable){
    LayersCombiner.setScatteringRefraction(enable);

    updateCmdFlags();
}

void deferredGraphics::updateCmdFlags(){
    std::fill(updateCommandBufferFlags.begin(), updateCommandBufferFlags.end(), true);
}

deferredGraphics& deferredGraphics::setEnableTransparentLayers(bool enable) {enableTransparentLayers = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableSkybox(bool enable)            {enableSkybox = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableBlur(bool enable)              {enableBlur = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableBloom(bool enable)             {enableBloom = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableSSLR(bool enable)              {enableSSLR = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableSSAO(bool enable)              {enableSSAO = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableScattering(bool enable)        {enableScattering = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableShadow(bool enable)            {enableShadow = enable; return *this;}
deferredGraphics& deferredGraphics::setEnableBoundingBox(bool enable)       {enableBoundingBox = enable; return *this;}
