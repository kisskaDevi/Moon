#include "deferredGraphics.h"
#include "operations.h"
#include "texture.h"
#include "node.h"
#include "model.h"
#include "light.h"
#include "object.h"
#include "camera.h"
#include "swapChain.h"
#include "depthMap.h"

#include "graphics.h"
#include "postProcessing.h"
#include "blur.h"
#include "bloom.h"
#include "sslr.h"
#include "ssao.h"
#include "layersCombiner.h"
#include "scattering.h"
#include "skybox.h"
#include "shadow.h"
#include "boundingBox.h"
#include "selector.h"

#include <cstring>

deferredGraphics::deferredGraphics(const std::filesystem::path& shadersPath, VkExtent2D extent, VkOffset2D offset, VkSampleCountFlagBits MSAASamples):
    shadersPath(shadersPath), extent(extent), offset(offset), MSAASamples(MSAASamples)
{
    enable["DeferredGraphics"] = true;
    enable["LayersCombiner"] = true;
    enable["PostProcessing"] = true;
    enable["Bloom"] = false;
    enable["Blur"] = false;
    enable["Skybox"] = false;
    enable["SSLR"] = false;
    enable["SSAO"] = false;
    enable["Shadow"] = false;
    enable["Scattering"] = false;
    enable["BoundingBox"] = false;
    enable["TransparentLayer"] = false;
    enable["Selector"] = false;

    link = &Link;
}

deferredGraphics::~deferredGraphics(){
    deferredGraphics::destroy();
}

void deferredGraphics::freeCommandBuffers(){
    CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::freeCommandBuffers ] commandPool is VK_NULL_HANDLE"));
    CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::freeCommandBuffers ] VkDevice is VK_NULL_HANDLE"));

    for(auto& [_,workflow]: workflows){
        workflow->freeCommandBuffer(commandPool);
    }
}

void deferredGraphics::destroyCommandPool(){
    if(commandPool){
        freeCommandBuffers();

        vkDestroyCommandPool(device.getLogical(), commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
}

void deferredGraphics::destroy(){
    for(auto& [_,map]: depthMaps){
        static_cast<shadowGraphics*>(workflows["Shadow"])->destroyFramebuffers(map);
        delete map;
        map = nullptr;
    }

    destroyCommandPool();

    for(auto& node: nodes){
        node->destroy(device.getLogical());
        delete node;
    }
    nodes.clear();

    for(auto& [_,texture] : emptyTextures){
        if(texture){
            texture->destroy(device.getLogical());
            texture = nullptr;
        }
    }

    for(auto& [_,workflow]: workflows){
        workflow->destroy();
        delete workflow;
    }
    workflows.clear();
    attachmentsMap.clear();

    Link.destroy();

    for (auto& buffer: storageBuffersHost){
        buffer.destroy(device.getLogical());
    }
    storageBuffersHost.clear();
    bufferMap.erase("storage");
}

void deferredGraphics::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK(vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool));
}

void deferredGraphics::create()
{
    createCommandPool();

    emptyTextures["black"] = ::createEmptyTexture(device, commandPool);
    emptyTextures["white"] = ::createEmptyTexture(device, commandPool, false);

    createGraphicsPasses();
    createCommandBuffers();
    updateDescriptorSets();

    for(auto& [lightSource,map]: depthMaps){
        depthMaps[lightSource] = new depthMap(device, commandPool, imageCount);
        if(lightSource->isShadowEnable() && enable["Shadow"]){
            static_cast<shadowGraphics*>(workflows["Shadow"])->createFramebuffers(depthMaps[lightSource]);
        }
    }
}

void deferredGraphics::createGraphicsPasses(){
    CHECK_M(commandPool == VK_NULL_HANDLE,       std::string("[ deferredGraphics::createGraphicsPasses ] VkCommandPool is VK_NULL_HANDLE"));
    CHECK_M(device.instance == VK_NULL_HANDLE,   std::string("[ deferredGraphics::createGraphicsPasses ] VkPhysicalDevice is VK_NULL_HANDLE"));
    CHECK_M(swapChainKHR == nullptr,             std::string("[ deferredGraphics::createGraphicsPasses ] swapChain is nullptr"));
    CHECK_M(cameraObject == nullptr,             std::string("[ deferredGraphics::createGraphicsPasses ] camera is nullptr"));

    workflows["DeferredGraphics"] = new graphics(enable["DeferredGraphics"], enable["TransparentLayer"], false, 0, &objects, &lights, &depthMaps);
    workflows["LayersCombiner"] = new layersCombiner(enable["LayersCombiner"], enable["TransparentLayer"] ? TransparentLayersCount : 0, true);
    workflows["PostProcessing"] = new postProcessingGraphics(enable["PostProcessing"]);
    workflows["Blur"] = new gaussianBlur(enable["Blur"]);
    workflows["Bloom"] = new bloomGraphics(enable["Bloom"], blitAttachmentsCount);
    workflows["Skybox"] = new skyboxGraphics(enable["Skybox"], &objects);
    workflows["SSLR"] = new SSLRGraphics(enable["SSLR"]);
    workflows["SSAO"] = new SSAOGraphics(enable["SSAO"]);
    workflows["Shadow"] = new shadowGraphics(enable["Shadow"], &objects, &depthMaps);
    workflows["Scattering"] = new scattering(enable["Scattering"], &lights, &depthMaps);
    workflows["BoundingBox"] = new boundingBoxGraphics(enable["BoundingBox"], &objects);
    for(uint32_t i = 0; i < TransparentLayersCount; i++){
        enable["TransparentLayer" + std::to_string(i)] = enable["TransparentLayer"];
        workflows["TransparentLayer" + std::to_string(i)] = new graphics(enable["TransparentLayer" + std::to_string(i)], enable["TransparentLayer"], true, i, &objects, &lights, &depthMaps);
    };
    workflows["Selector"] = new selectorGraphics(enable["Selector"]);


    for(auto& [_,workflow]: workflows){
        imageInfo info{imageCount, swapChainKHR->getFormat(), VkOffset2D{0,0}, extent, extent, MSAASamples};

        workflow->setEmptyTexture(emptyTextures);
        workflow->setShadersPath(shadersPath);
        workflow->setDeviceProp(device.instance, device.getLogical());
        workflow->setImageProp(&info);
    }

    imageInfo scatterInfo{imageCount, VK_FORMAT_R32G32B32A32_SFLOAT, VkOffset2D{0,0}, extent, extent, MSAASamples};
    workflows["Scattering"]->setImageProp(&scatterInfo);

    imageInfo shadowsInfo{imageCount,VK_FORMAT_D32_SFLOAT,VkOffset2D{0,0},VkExtent2D{1024,1024},VkExtent2D{1024,1024},MSAASamples};
    workflows["Shadow"]->setImageProp(&shadowsInfo);

    imageInfo swapChainInfo{imageCount, swapChainKHR->getFormat(), offset, extent, swapChainKHR->getExtent(), MSAASamples};
    workflows["PostProcessing"]->setImageProp(&swapChainInfo);

    for(auto& [_,workflow]: workflows){
        workflow->create(attachmentsMap);
    }

    Link.setShadersPath(shadersPath);
    Link.setDeviceProp(device.getLogical());
    Link.setImageCount(imageCount);
    Link.createDescriptorSetLayout();
    Link.createPipeline(&swapChainInfo);
    Link.createDescriptorPool();
    Link.createDescriptorSets();

    createStorageBuffers(imageCount);

    updateCommandBufferFlags.resize(imageCount, true);
}

void deferredGraphics::updateDescriptorSets(){
    CHECK_M(cameraObject == nullptr, std::string("[ deferredGraphics::updateDescriptorSets ] camera is nullptr"));

    for(auto& [name, workflow]: workflows){
        workflow->updateDescriptorSets(bufferMap, attachmentsMap);
    }
    Link.updateDescriptorSets(attachmentsMap["final"].second.front());
}

void deferredGraphics::createCommandBuffers(){
    CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::createCommandBuffers ] VkCommandPool is VK_NULL_HANDLE"));

    for(auto& [_,workflow]: workflows){
        workflow->createCommandBuffers(commandPool);
    }

    copyCommandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = imageCount;
    CHECK(vkAllocateCommandBuffers(device.getLogical(), &allocInfo, copyCommandBuffers.data()));

    updateCmdFlags();

    auto getTransparentLayersCommandBuffers = [this](uint32_t imageIndex) -> std::vector<VkCommandBuffer>{
        std::vector<VkCommandBuffer> commandBuffers;
        for(uint32_t i = 0; i < TransparentLayersCount; i++){
            commandBuffers.push_back(workflows["TransparentLayer" + std::to_string(i)]->getCommandBuffer(imageIndex));
        }
        return commandBuffers;
    };

    nodes.resize(imageCount);
    for(uint32_t imageIndex = 0; imageIndex < imageCount; imageIndex++){
        nodes[imageIndex]
         = new node({
            stage(  {copyCommandBuffers[imageIndex]}, VK_PIPELINE_STAGE_TRANSFER_BIT, device.getQueue(0,0))
        }, new node({
            stage(  {workflows["Shadow"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device.getQueue(0,0)),
            stage(  {workflows["Skybox"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device.getQueue(0,0))
        }, new node({
            stage(  {workflows["DeferredGraphics"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device.getQueue(0,0)),
            stage(  getTransparentLayersCommandBuffers(imageIndex), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device.getQueue(0,0))
        }, new node({
            stage(  {workflows["Scattering"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device.getQueue(0,0))
        }, new node({
            stage(  {workflows["LayersCombiner"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device.getQueue(0,0))
        }, new node({
            stage(  {workflows["Selector"]->getCommandBuffer(imageIndex),
                     workflows["SSLR"]->getCommandBuffer(imageIndex),
                     workflows["SSAO"]->getCommandBuffer(imageIndex),
                     workflows["Bloom"]->getCommandBuffer(imageIndex),
                     workflows["Blur"]->getCommandBuffer(imageIndex),
                     workflows["BoundingBox"]->getCommandBuffer(imageIndex),
                     workflows["PostProcessing"]->getCommandBuffer(imageIndex)},
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device.getQueue(0,0))
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

void deferredGraphics::update(uint32_t imageIndex) {
    updateBuffers(imageIndex);
    updateCommandBuffer(imageIndex);
}

void deferredGraphics::updateCommandBuffer(uint32_t imageIndex){
    if(updateCommandBufferFlags[imageIndex]){
        for(auto& [name, workflow]: workflows){
            workflow->beginCommandBuffer(imageIndex);
            workflow->updateCommandBuffer(imageIndex);
            workflow->endCommandBuffer(imageIndex);
        }
        updateCommandBufferFlags[imageIndex] = false;
    }
}

void deferredGraphics::updateBuffers(uint32_t imageIndex){
    CHECK(vkResetCommandBuffer(copyCommandBuffers[imageIndex],0));

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;
    CHECK(vkBeginCommandBuffer(copyCommandBuffers[imageIndex], &beginInfo));

    if(cameraObject){
        cameraObject->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for(auto& object: objects){
        object->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for(auto& light: lights){
        light->update(imageIndex, copyCommandBuffers[imageIndex]);
    }

    vkEndCommandBuffer(copyCommandBuffers[imageIndex]);
}

void deferredGraphics::createStorageBuffers(uint32_t imageCount){
    storageBuffersHost.resize(imageCount);
    bufferMap["storage"] = {sizeof(StorageBufferObject),{}};
    for (auto& buffer: storageBuffersHost){
        Buffer::create( device.instance,
                        device.getLogical(),
                        sizeof(StorageBufferObject),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
        CHECK(vkMapMemory(device.getLogical(), buffer.memory, 0, sizeof(StorageBufferObject), 0, &buffer.map));
        bufferMap["storage"].second.push_back(buffer.instance);
    }
}

void deferredGraphics::updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey){
    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = vector<float,4>(mousex,mousey,0.0f,0.0f);
        StorageUBO.number = std::numeric_limits<uint32_t>::max();
        StorageUBO.depth = 1.0f;
    std::memcpy(storageBuffersHost[currentImage].map, &StorageUBO, sizeof(StorageUBO));
}

void deferredGraphics::readStorageBuffer(uint32_t currentImage, uint32_t& primitiveNumber, float& depth){
    StorageBufferObject storageBuffer{};
    std::memcpy((void*)&storageBuffer, (void*)storageBuffersHost[currentImage].map, sizeof(StorageBufferObject));
    primitiveNumber = storageBuffer.number;
    depth = storageBuffer.depth;
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
    bufferMap["camera"] = {cameraObject->getBufferRange(),{}};
    for(uint32_t i = 0; i < imageCount; i++){
        bufferMap["camera"].second.push_back(cameraObject->getBuffer(i));
    }
}

void deferredGraphics::remove(camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject->destroy(device.getLogical());
        this->cameraObject = nullptr;
        bufferMap.erase("camera");
    }
}

void deferredGraphics::bind(light* lightSource){
    if(depthMaps.count(lightSource) == 0){
        depthMaps[lightSource] = new depthMap(device, commandPool, imageCount);
        if(lightSource->isShadowEnable() && enable["Shadow"]){
            static_cast<shadowGraphics*>(workflows["Shadow"])->createFramebuffers(depthMaps[lightSource]);
        }
    }
    lightSource->create(device, commandPool, imageCount);
    lights.push_back(lightSource);

    updateCmdFlags();
}

bool deferredGraphics::remove(light* lightSource){
    size_t size = lights.size();
    lightSource->destroy(device.getLogical());
    lights.erase(std::remove(lights.begin(), lights.end(), lightSource), lights.end());

    if(depthMaps.count(lightSource)){
        static_cast<shadowGraphics*>(workflows["Shadow"])->destroyFramebuffers(depthMaps[lightSource]);
        delete depthMaps[lightSource];
        depthMaps.erase(lightSource);
    }

    updateCmdFlags();
    return size - objects.size() > 0;
}

void deferredGraphics::bind(object* object){
    object->create(device, commandPool, imageCount);
    objects.push_back(object);
    updateCmdFlags();
}

bool deferredGraphics::remove(object* object){
    size_t size = objects.size();
    object->destroy(device.getLogical());
    objects.erase(std::remove(objects.begin(), objects.end(), object), objects.end());
    updateCmdFlags();
    return size - objects.size() > 0;
}

void deferredGraphics::updateCmdFlags(){
    std::fill(updateCommandBufferFlags.begin(), updateCommandBufferFlags.end(), true);
}

deferredGraphics& deferredGraphics::setEnable(const std::string& name, bool enable){
    this->enable[name] = enable;
    return *this;
}

bool deferredGraphics::getEnable(const std::string& name){
    return enable[name];
}

deferredGraphics& deferredGraphics::setMinAmbientFactor(const float& minAmbientFactor){
    static_cast<graphics*>(workflows["DeferredGraphics"])->setMinAmbientFactor(minAmbientFactor);
    for(uint32_t i = 0; i < TransparentLayersCount; i++){
        static_cast<graphics*>(workflows["TransparentLayer" + std::to_string(i)])->setMinAmbientFactor(minAmbientFactor);
    }

    updateCmdFlags();
    return *this;
}

deferredGraphics& deferredGraphics::setScatteringRefraction(bool enable){
    static_cast<layersCombiner*>(workflows["LayersCombiner"])->setScatteringRefraction(enable);

    updateCmdFlags();
    return *this;
}

deferredGraphics& deferredGraphics::setBlitFactor(float blitFactor){
    if(enable["Bloom"] && blitFactor >= 1.0f){
        static_cast<bloomGraphics*>(workflows["Bloom"])->setBlitFactor(blitFactor).setSamplerStepX(blitFactor).setSamplerStepY(blitFactor);
        updateCmdFlags();
    }
    return *this;
}

deferredGraphics& deferredGraphics::setBlurDepth(float blurDepth){
    static_cast<gaussianBlur*>(workflows["Blur"])->setBlurDepth(blurDepth);
    static_cast<layersCombiner*>(workflows["LayersCombiner"])->setBlurDepth(blurDepth);
    updateCmdFlags();
    return *this;
}

deferredGraphics& deferredGraphics::setExtentAndOffset(VkExtent2D extent, VkOffset2D offset) {
    this->offset = offset;
    this->extent = extent;
    return *this;
}

deferredGraphics& deferredGraphics::setShadersPath(const std::filesystem::path& path){
    shadersPath = path;
    return *this;
}
