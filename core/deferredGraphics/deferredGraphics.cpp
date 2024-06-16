#include "deferredGraphics.h"
#include "operations.h"
#include "texture.h"
#include "node.h"
#include "model.h"
#include "light.h"
#include "object.h"
#include "camera.h"
#include "depthMap.h"
#include "swapChain.h"

#include "link.h"
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

namespace moon::deferredGraphics {

DeferredGraphics::DeferredGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples):
    shadersPath(shadersPath), workflowsShadersPath(workflowsShadersPath), extent(extent), MSAASamples(MSAASamples)
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

    deferredLink = new Link;
    link = deferredLink;
}

DeferredGraphics::~DeferredGraphics(){
    DeferredGraphics::destroy();
    delete deferredLink;
}

void DeferredGraphics::freeCommandBuffers(){
    CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ DeferredGraphics::freeCommandBuffers ] commandPool is VK_NULL_HANDLE"));
    CHECK_M(device->getLogical() == VK_NULL_HANDLE, std::string("[ DeferredGraphics::freeCommandBuffers ] VkDevice is VK_NULL_HANDLE"));

    for(auto& [_,workflow]: workflows){
        workflow->freeCommandBuffer(commandPool);
    }
}

void DeferredGraphics::destroyCommandPool(){
    if(commandPool){
        freeCommandBuffers();

        vkDestroyCommandPool(device->getLogical(), commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
}

void DeferredGraphics::destroy(){
    for(auto& [_,map]: depthMaps){
        static_cast<moon::workflows::ShadowGraphics*>(workflows["Shadow"])->destroyFramebuffers(map);
        delete map;
        map = nullptr;
    }

    destroyCommandPool();

    for(auto& node: nodes){
        node->destroy(device->getLogical());
        delete node;
    }
    nodes.clear();

    for(auto& [_,texture] : emptyTextures){
        if(texture){
            texture->destroy(device->getLogical());
            texture = nullptr;
        }
    }

    for(auto& [_,workflow]: workflows){
        workflow->destroy();
        delete workflow;
    }
    workflows.clear();
    aDatabase.destroy();

    deferredLink->destroy();

    storageBuffersHost.destroy(device->getLogical());
    bDatabase.buffersMap.erase("storage");
}

void DeferredGraphics::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK(vkCreateCommandPool(device->getLogical(), &poolInfo, nullptr, &commandPool));
}

void DeferredGraphics::create()
{
    createCommandPool();

    emptyTextures["black"] = moon::utils::createEmptyTexture(*device, commandPool);
    emptyTextures["white"] = moon::utils::createEmptyTexture(*device, commandPool, false);
    aDatabase.addEmptyTexture("black", emptyTextures["black"]);
    aDatabase.addEmptyTexture("white", emptyTextures["white"]);

    createGraphicsPasses();
    createCommandBuffers();
    updateDescriptorSets();

    for(auto& [lightSource,map]: depthMaps){
        depthMaps[lightSource] = new moon::utils::DepthMap(*device, commandPool, imageCount);
        if(lightSource->isShadowEnable() && enable["Shadow"]){
            static_cast<moon::workflows::ShadowGraphics*>(workflows["Shadow"])->createFramebuffers(depthMaps[lightSource]);
        }
    }
}

void DeferredGraphics::createGraphicsPasses(){
    CHECK_M(commandPool == VK_NULL_HANDLE,       std::string("[ DeferredGraphics::createGraphicsPasses ] VkCommandPool is VK_NULL_HANDLE"));
    CHECK_M(device->instance == VK_NULL_HANDLE,   std::string("[ DeferredGraphics::createGraphicsPasses ] VkPhysicalDevice is VK_NULL_HANDLE"));
    CHECK_M(cameraObject == nullptr,             std::string("[ DeferredGraphics::createGraphicsPasses ] camera is nullptr"));

    GraphicsParameters graphicsParams;
    graphicsParams.in.camera = "camera";
    graphicsParams.out.image = "image";
    graphicsParams.out.blur = "blur";
    graphicsParams.out.bloom = "bloom";
    graphicsParams.out.position = "GBuffer.position";
    graphicsParams.out.normal = "GBuffer.normal";
    graphicsParams.out.color = "GBuffer.color";
    graphicsParams.out.depth = "GBuffer.depth";
    graphicsParams.out.transparency = "transparency";

    moon::workflows::SkyboxParameters skyboxParams;
    skyboxParams.in.camera = graphicsParams.in.camera;
    skyboxParams.out.baseColor = "skybox.color";
    skyboxParams.out.bloom = "skybox.bloom";

    moon::workflows::ScatteringParameters scatteringParams;
    scatteringParams.in.camera = graphicsParams.in.camera;
    scatteringParams.in.depth = graphicsParams.out.depth;
    scatteringParams.out.scattering = "scattering";

    moon::workflows::SSLRParameters SSLRParams;
    SSLRParams.in.camera = graphicsParams.in.camera;
    SSLRParams.in.position = graphicsParams.out.position;
    SSLRParams.in.normal = graphicsParams.out.normal;
    SSLRParams.in.color = graphicsParams.out.image;
    SSLRParams.in.depth = graphicsParams.out.depth;
    SSLRParams.in.firstTransparency = graphicsParams.out.transparency + "0";
    SSLRParams.in.defaultDepthTexture = "white";
    SSLRParams.out.sslr = "sslr";

    LayersCombinerParameters layersCombinerParams;
    layersCombinerParams.in.camera = graphicsParams.in.camera;
    layersCombinerParams.in.color = graphicsParams.out.image;
    layersCombinerParams.in.bloom = graphicsParams.out.bloom;
    layersCombinerParams.in.position = graphicsParams.out.position;
    layersCombinerParams.in.normal = graphicsParams.out.normal;
    layersCombinerParams.in.depth = graphicsParams.out.depth;
    layersCombinerParams.in.skyboxColor = skyboxParams.out.baseColor;
    layersCombinerParams.in.skyboxBloom = skyboxParams.out.bloom;
    layersCombinerParams.in.scattering = scatteringParams.out.scattering;
    layersCombinerParams.in.sslr = SSLRParams.out.sslr;
    layersCombinerParams.in.transparency = graphicsParams.out.transparency;
    layersCombinerParams.in.defaultDepthTexture = "white";
    layersCombinerParams.out.color = "combined.color";
    layersCombinerParams.out.bloom = "combined.bloom";
    layersCombinerParams.out.blur = "combined.blur";

    moon::workflows::BloomParameters bloomParams;
    bloomParams.in.bloom = layersCombinerParams.out.bloom;
    bloomParams.out.bloom = "bloomFinal";

    moon::workflows::GaussianBlurParameters blurParams;
    blurParams.in.blur = layersCombinerParams.out.blur;
    blurParams.out.blur = "blured";

    moon::workflows::BoundingBoxParameters bbParams;
    bbParams.in.camera = graphicsParams.in.camera;
    bbParams.out.boundingBox = "boundingBox";

    moon::workflows::SSAOParameters SSAOParams;
    SSAOParams.in.camera = graphicsParams.in.camera;
    SSAOParams.in.position = graphicsParams.out.position;
    SSAOParams.in.normal = graphicsParams.out.normal;
    SSAOParams.in.color = graphicsParams.out.image;
    SSAOParams.in.depth = graphicsParams.out.depth;
    SSAOParams.in.defaultDepthTexture = "white";
    SSAOParams.out.ssao = "ssao";

    moon::workflows::PostProcessingParameters postProcessingParams;
    postProcessingParams.in.baseColor = layersCombinerParams.out.color;
    postProcessingParams.in.bloom = bloomParams.out.bloom;
    postProcessingParams.in.blur = blurParams.out.blur;
    postProcessingParams.in.boundingBox = bbParams.out.boundingBox;
    postProcessingParams.in.ssao = SSAOParams.out.ssao;
    postProcessingParams.out.postProcessing = "final";

    moon::workflows::SelectorParameters selectorParams;
    selectorParams.in.storageBuffer = "storage";
    selectorParams.in.position = graphicsParams.out.position;
    selectorParams.in.depth = graphicsParams.out.depth;
    selectorParams.in.transparency = graphicsParams.out.transparency;
    selectorParams.in.defaultDepthTexture = "white";
    selectorParams.out.selector = "selector";

    workflows["DeferredGraphics"] = new Graphics(graphicsParams, enable["DeferredGraphics"], enable["TransparentLayer"], false, 0, &objects, &lights, &depthMaps);
    workflows["DeferredGraphics"]->setShadersPath(shadersPath);

    workflows["LayersCombiner"] = new LayersCombiner(layersCombinerParams, enable["LayersCombiner"], enable["TransparentLayer"] ? TransparentLayersCount : 0, true);
    workflows["LayersCombiner"]->setShadersPath(shadersPath);

    workflows["PostProcessing"] = new moon::workflows::PostProcessingGraphics(postProcessingParams, enable["PostProcessing"]);
    workflows["PostProcessing"]->setShadersPath(shadersPath);

    for(uint32_t i = 0; i < TransparentLayersCount; i++){
        const auto key = "TransparentLayer" + std::to_string(i);
        enable[key] = enable["TransparentLayer"];
        workflows[key] = new Graphics(graphicsParams, enable["TransparentLayer" + std::to_string(i)], enable["TransparentLayer"], true, i, &objects, &lights, &depthMaps);
        workflows[key]->setShadersPath(shadersPath);
    };

    workflows["Blur"] = new moon::workflows::GaussianBlur(blurParams, enable["Blur"]);
    workflows["Blur"]->setShadersPath(workflowsShadersPath);
    workflows["Bloom"] = new moon::workflows::BloomGraphics(bloomParams, enable["Bloom"], blitAttachmentsCount);
    workflows["Bloom"]->setShadersPath(workflowsShadersPath);
    workflows["Skybox"] = new moon::workflows::SkyboxGraphics(skyboxParams, enable["Skybox"], &objects);
    workflows["Skybox"]->setShadersPath(workflowsShadersPath);
    workflows["SSLR"] = new moon::workflows::SSLRGraphics(SSLRParams, enable["SSLR"]);
    workflows["SSLR"]->setShadersPath(workflowsShadersPath);
    workflows["SSAO"] = new moon::workflows::SSAOGraphics(SSAOParams, enable["SSAO"]);
    workflows["SSAO"]->setShadersPath(workflowsShadersPath);
    workflows["Shadow"] = new moon::workflows::ShadowGraphics(enable["Shadow"], &objects, &depthMaps);
    workflows["Shadow"]->setShadersPath(workflowsShadersPath);
    workflows["Scattering"] = new moon::workflows::Scattering(scatteringParams, enable["Scattering"], &lights, &depthMaps);
    workflows["Scattering"]->setShadersPath(workflowsShadersPath);
    workflows["BoundingBox"] = new moon::workflows::BoundingBoxGraphics(bbParams, enable["BoundingBox"], &objects);
    workflows["BoundingBox"]->setShadersPath(workflowsShadersPath);
    workflows["Selector"] = new moon::workflows::SelectorGraphics(selectorParams, enable["Selector"]);
    workflows["Selector"]->setShadersPath(workflowsShadersPath);


    for(auto& [_,workflow]: workflows){
        moon::utils::ImageInfo info{imageCount, format, extent, MSAASamples};
        workflow->setDeviceProp(device->instance, device->getLogical());
        workflow->setImageProp(&info);
    }

    moon::utils::ImageInfo scatterInfo{imageCount, VK_FORMAT_R32G32B32A32_SFLOAT, extent, MSAASamples};
    workflows["Scattering"]->setImageProp(&scatterInfo);

    moon::utils::ImageInfo shadowsInfo{imageCount,VK_FORMAT_D32_SFLOAT,VkExtent2D{1024,1024},MSAASamples};
    workflows["Shadow"]->setImageProp(&shadowsInfo);

    moon::utils::ImageInfo postProcessingInfo{imageCount, format, extent, MSAASamples};
    workflows["PostProcessing"]->setImageProp(&postProcessingInfo);

    for(auto& [_,workflow]: workflows){
        workflow->create(aDatabase);
    }

    moon::utils::ImageInfo linkInfo{imageCount, format, swapChainKHR->getExtent(), MSAASamples};
    deferredLink->setShadersPath(shadersPath);
    deferredLink->setDeviceProp(device->getLogical());
    deferredLink->setImageCount(imageCount);
    deferredLink->createDescriptorSetLayout();
    deferredLink->createPipeline(&linkInfo);
    deferredLink->createDescriptorPool();
    deferredLink->createDescriptorSets();

    createStorageBuffers(imageCount);

    updateCommandBufferFlags.resize(imageCount, true);
}

void DeferredGraphics::updateDescriptorSets(){
    CHECK_M(cameraObject == nullptr, std::string("[ deferredGraphics::updateDescriptorSets ] camera is nullptr"));

    for(auto& [name, workflow]: workflows){
        workflow->updateDescriptorSets(bDatabase, aDatabase);
    }
    deferredLink->updateDescriptorSets(aDatabase.get("final"));
}

void DeferredGraphics::createCommandBuffers(){
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
    CHECK(vkAllocateCommandBuffers(device->getLogical(), &allocInfo, copyCommandBuffers.data()));

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
         = new moon::utils::Node({
            moon::utils::Stage(  {copyCommandBuffers[imageIndex]}, VK_PIPELINE_STAGE_TRANSFER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node({
            moon::utils::Stage(  {workflows["Shadow"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  {workflows["Skybox"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0))
        }, new moon::utils::Node({
            moon::utils::Stage(  {workflows["DeferredGraphics"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  getTransparentLayersCommandBuffers(imageIndex), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node({
            moon::utils::Stage(  {workflows["Scattering"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  {workflows["SSLR"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node({
            moon::utils::Stage(  {workflows["LayersCombiner"]->getCommandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node({
            moon::utils::Stage(  {workflows["Selector"]->getCommandBuffer(imageIndex),
                     workflows["SSAO"]->getCommandBuffer(imageIndex),
                     workflows["Bloom"]->getCommandBuffer(imageIndex),
                     workflows["Blur"]->getCommandBuffer(imageIndex),
                     workflows["BoundingBox"]->getCommandBuffer(imageIndex),
                     workflows["PostProcessing"]->getCommandBuffer(imageIndex)},
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, nullptr))))));

        nodes[imageIndex]->createSemaphores(device->getLogical());
    }
}

std::vector<std::vector<VkSemaphore>> DeferredGraphics::submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex){
    if(externalSemaphore.size()){
        nodes[imageIndex]->setExternalSemaphore(externalSemaphore);
    }
    if(externalFence.size()){
        nodes[imageIndex]->back()->setExternalFence(externalFence);
    }

    nodes[imageIndex]->submit();

    return nodes[imageIndex]->back()->getBackSemaphores();
}

void DeferredGraphics::update(uint32_t imageIndex) {
    updateBuffers(imageIndex);
    updateCommandBuffer(imageIndex);
}

void DeferredGraphics::setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size) {
    this->offset = offset;
    this->size = size;
    deferredLink->setPositionInWindow(offset, size);
}

void DeferredGraphics::updateCommandBuffer(uint32_t imageIndex){
    if(updateCommandBufferFlags[imageIndex]){
        for(auto& [name, workflow]: workflows){
            workflow->beginCommandBuffer(imageIndex);
            workflow->updateCommandBuffer(imageIndex);
            workflow->endCommandBuffer(imageIndex);
        }
        updateCommandBufferFlags[imageIndex] = false;
    }
}

void DeferredGraphics::updateBuffers(uint32_t imageIndex){
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

void DeferredGraphics::createStorageBuffers(uint32_t imageCount){
    storageBuffersHost.create(device->instance,
                              device->getLogical(),
                              sizeof(StorageBufferObject),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              imageCount);
    storageBuffersHost.map(device->getLogical());
    bDatabase.addBufferData("storage", &storageBuffersHost);
}

void DeferredGraphics::updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey){
    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = moon::math::Vector<float,4>(mousex,mousey,0.0f,0.0f);
        StorageUBO.number = std::numeric_limits<uint32_t>::max();
        StorageUBO.depth = 1.0f;
    std::memcpy(storageBuffersHost.instances[currentImage].map, &StorageUBO, sizeof(StorageUBO));
}

void DeferredGraphics::readStorageBuffer(uint32_t currentImage, uint32_t& primitiveNumber, float& depth){
    StorageBufferObject storageBuffer{};
    std::memcpy((void*)&storageBuffer, (void*)storageBuffersHost.instances[currentImage].map, sizeof(StorageBufferObject));
    primitiveNumber = storageBuffer.number;
    depth = storageBuffer.depth;
}

void DeferredGraphics::create(moon::interfaces::Model *pModel){
    pModel->create(*device, commandPool);
}

void DeferredGraphics::destroy(moon::interfaces::Model* pModel){
    pModel->destroy(device->getLogical());
}

void DeferredGraphics::bind(moon::interfaces::Camera* cameraObject){
    this->cameraObject = cameraObject;
    cameraObject->create(*device, imageCount);
    bDatabase.addBufferData("camera", &cameraObject->getBuffers());
}

void DeferredGraphics::remove(moon::interfaces::Camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject->destroy(device->getLogical());
        this->cameraObject = nullptr;
        bDatabase.buffersMap.erase("camera");
    }
}

void DeferredGraphics::bind(moon::interfaces::Light* lightSource){
    if(depthMaps.count(lightSource) == 0){
        depthMaps[lightSource] = new moon::utils::DepthMap(*device, commandPool, imageCount);
        if(lightSource->isShadowEnable() && enable["Shadow"]){
            static_cast<moon::workflows::ShadowGraphics*>(workflows["Shadow"])->createFramebuffers(depthMaps[lightSource]);
        }
    }
    lightSource->create(*device, commandPool, imageCount);
    lights.push_back(lightSource);

    updateCmdFlags();
}

bool DeferredGraphics::remove(moon::interfaces::Light* lightSource){
    size_t size = lights.size();
    lightSource->destroy(device->getLogical());
    lights.erase(std::remove(lights.begin(), lights.end(), lightSource), lights.end());

    if(depthMaps.count(lightSource)){
        static_cast<moon::workflows::ShadowGraphics*>(workflows["Shadow"])->destroyFramebuffers(depthMaps[lightSource]);
        delete depthMaps[lightSource];
        depthMaps.erase(lightSource);
    }

    updateCmdFlags();
    return size - objects.size() > 0;
}

void DeferredGraphics::bind(moon::interfaces::Object* object){
    object->create(*device, commandPool, imageCount);
    objects.push_back(object);
    updateCmdFlags();
}

bool DeferredGraphics::remove(moon::interfaces::Object* object){
    size_t size = objects.size();
    object->destroy(device->getLogical());
    objects.erase(std::remove(objects.begin(), objects.end(), object), objects.end());
    updateCmdFlags();
    return size - objects.size() > 0;
}

void DeferredGraphics::updateCmdFlags(){
    std::fill(updateCommandBufferFlags.begin(), updateCommandBufferFlags.end(), true);
}

DeferredGraphics& DeferredGraphics::setEnable(const std::string& name, bool enable){
    this->enable[name] = enable;
    return *this;
}

bool DeferredGraphics::getEnable(const std::string& name){
    return enable[name];
}

DeferredGraphics& DeferredGraphics::setMinAmbientFactor(const float& minAmbientFactor){
    static_cast<Graphics*>(workflows["DeferredGraphics"])->setMinAmbientFactor(minAmbientFactor);
    for(uint32_t i = 0; i < TransparentLayersCount; i++){
        static_cast<Graphics*>(workflows["TransparentLayer" + std::to_string(i)])->setMinAmbientFactor(minAmbientFactor);
    }

    updateCmdFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setScatteringRefraction(bool enable){
    static_cast<LayersCombiner*>(workflows["LayersCombiner"])->setScatteringRefraction(enable);

    updateCmdFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setBlitFactor(float blitFactor){
    if(enable["Bloom"] && blitFactor >= 1.0f){
        static_cast<moon::workflows::BloomGraphics*>(workflows["Bloom"])->setBlitFactor(blitFactor).setSamplerStepX(blitFactor).setSamplerStepY(blitFactor);
        updateCmdFlags();
    }
    return *this;
}

DeferredGraphics& DeferredGraphics::setBlurDepth(float blurDepth){
    static_cast<moon::workflows::GaussianBlur*>(workflows["Blur"])->setBlurDepth(blurDepth);
    static_cast<LayersCombiner*>(workflows["LayersCombiner"])->setBlurDepth(blurDepth);
    updateCmdFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setExtent(VkExtent2D extent) {
    this->extent = extent;
    return *this;
}

DeferredGraphics& DeferredGraphics::setShadersPath(const std::filesystem::path& path){
    shadersPath = path;
    return *this;
}

}
