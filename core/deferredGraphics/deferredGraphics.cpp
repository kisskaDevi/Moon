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

    link = &deferredLink;
}

void DeferredGraphics::create()
{
    CHECK(commandPool.create(device->getLogical()));

    aDatabase.destroy();
    emptyTextures["black"] = moon::utils::Texture::empty(*device, commandPool);
    emptyTextures["white"] = moon::utils::Texture::empty(*device, commandPool, false);
    aDatabase.addEmptyTexture("black", &emptyTextures["black"]);
    aDatabase.addEmptyTexture("white", &emptyTextures["white"]);

    createGraphicsPasses();
    createCommandBuffers();
    updateDescriptorSets();
}

void DeferredGraphics::createGraphicsPasses(){
    CHECK_M(!commandPool,                        std::string("[ DeferredGraphics::createGraphicsPasses ] VkCommandPool is VK_NULL_HANDLE"));
    CHECK_M(device->instance == VK_NULL_HANDLE,  std::string("[ DeferredGraphics::createGraphicsPasses ] VkPhysicalDevice is VK_NULL_HANDLE"));
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

    moon::utils::ImageInfo info{ imageCount, format, extent, MSAASamples };
    moon::utils::ImageInfo scatterInfo{ imageCount, VK_FORMAT_R32G32B32A32_SFLOAT, extent, MSAASamples };
    moon::utils::ImageInfo shadowsInfo{ imageCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, MSAASamples };

    workflows.clear();

    workflows["DeferredGraphics"] = std::make_unique<Graphics>(info, shadersPath, graphicsParams, enable["DeferredGraphics"], enable["TransparentLayer"], false, 0, &objects, &lights, &depthMaps);
    workflows["LayersCombiner"] = std::make_unique<LayersCombiner>(info, shadersPath, layersCombinerParams, enable["LayersCombiner"], enable["TransparentLayer"] ? transparentLayersCount : 0, true);
    workflows["PostProcessing"] = std::make_unique<moon::workflows::PostProcessingGraphics>(info, workflowsShadersPath, postProcessingParams, enable["PostProcessing"]);

    for(uint32_t i = 0; i < transparentLayersCount; i++){
        const auto key = "TransparentLayer" + std::to_string(i);
        enable[key] = enable["TransparentLayer"];
        workflows[key] = std::make_unique<Graphics>(info, shadersPath, graphicsParams, enable[key], enable["TransparentLayer"], true, i, &objects, &lights, &depthMaps);
    };

    workflows["Blur"] = std::make_unique<moon::workflows::GaussianBlur>(info, workflowsShadersPath, blurParams, enable["Blur"]);
    workflows["Bloom"] = std::make_unique<moon::workflows::BloomGraphics>(info, workflowsShadersPath, bloomParams, enable["Bloom"], blitAttachmentsCount);
    workflows["Skybox"] = std::make_unique<moon::workflows::SkyboxGraphics>(info, workflowsShadersPath, skyboxParams, enable["Skybox"], &objects);
    workflows["SSLR"] = std::make_unique<moon::workflows::SSLRGraphics>(info, workflowsShadersPath, SSLRParams, enable["SSLR"]);
    workflows["SSAO"] = std::make_unique<moon::workflows::SSAOGraphics>(info, workflowsShadersPath, SSAOParams, enable["SSAO"]);
    workflows["Shadow"] = std::make_unique<moon::workflows::ShadowGraphics>(shadowsInfo, workflowsShadersPath, enable["Shadow"], &objects, &depthMaps);
    workflows["Scattering"] = std::make_unique<moon::workflows::Scattering>(scatterInfo, workflowsShadersPath, scatteringParams, enable["Scattering"], &lights, &depthMaps);
    workflows["BoundingBox"] = std::make_unique<moon::workflows::BoundingBoxGraphics>(info, workflowsShadersPath, bbParams, enable["BoundingBox"], &objects);
    workflows["Selector"] = std::make_unique<moon::workflows::SelectorGraphics>(info, workflowsShadersPath, selectorParams, enable["Selector"]);

    for(auto& [_,workflow]: workflows){
        workflow->setDeviceProp(device->instance, device->getLogical());
        workflow->create(aDatabase);
    }

    moon::utils::ImageInfo linkInfo{imageCount, format, swapChainKHR->info().Extent, MSAASamples};
    deferredLink.setShadersPath(shadersPath);
    deferredLink.setDeviceProp(device->getLogical());
    deferredLink.setImageCount(imageCount);
    deferredLink.createDescriptorSetLayout();
    deferredLink.createPipeline(&linkInfo);
    deferredLink.createDescriptorPool();

    createStorageBuffers(imageCount);
}

void DeferredGraphics::updateDescriptorSets(){
    for(auto& [name, workflow]: workflows){
        workflow->updateDescriptorSets(bDatabase, aDatabase);
    }
    deferredLink.updateDescriptorSets(aDatabase.get("final"));
}

void DeferredGraphics::createCommandBuffers(){
    CHECK_M(!commandPool, std::string("[ deferredGraphics::createCommandBuffers ] VkCommandPool is VK_NULL_HANDLE"));

    for(auto& [_,workflow]: workflows){
        workflow->createCommandBuffers(commandPool);
    }

    copyCommandBuffers = commandPool.allocateCommandBuffers(imageCount);

    auto getTransparentLayersCommandBuffers = [this](uint32_t imageIndex) -> std::vector<VkCommandBuffer>{
        std::vector<VkCommandBuffer> commandBuffers;
        for(uint32_t i = 0; i < transparentLayersCount; i++){
            commandBuffers.push_back(workflows["TransparentLayer" + std::to_string(i)]->commandBuffer(imageIndex));
        }
        return commandBuffers;
    };

    nodes.resize(imageCount);
    for(uint32_t imageIndex = 0; imageIndex < imageCount; imageIndex++){
        nodes[imageIndex] = moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {copyCommandBuffers[imageIndex]}, VK_PIPELINE_STAGE_TRANSFER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {workflows["Shadow"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  {workflows["Skybox"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0))
        }, new moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {workflows["DeferredGraphics"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  getTransparentLayersCommandBuffers(imageIndex), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {workflows["Scattering"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  {workflows["SSLR"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {workflows["LayersCombiner"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {workflows["Selector"]->commandBuffer(imageIndex),
                                  workflows["SSAO"]->commandBuffer(imageIndex),
                                  workflows["Bloom"]->commandBuffer(imageIndex),
                                  workflows["Blur"]->commandBuffer(imageIndex),
                                  workflows["BoundingBox"]->commandBuffer(imageIndex),
                                  workflows["PostProcessing"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, nullptr))))));

        nodes[imageIndex].createSemaphores();
    }
}

std::vector<std::vector<VkSemaphore>> DeferredGraphics::submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex){
    if(externalSemaphore.size()){
        nodes[imageIndex].setExternalSemaphore(externalSemaphore);
    }
    if(externalFence.size()){
        nodes[imageIndex].back()->setExternalFence(externalFence);
    }

    nodes[imageIndex].submit();

    return nodes[imageIndex].back()->getBackSemaphores();
}

void DeferredGraphics::update(uint32_t imageIndex) {
    updateBuffers(imageIndex);
    updateCommandBuffer(imageIndex);
}

void DeferredGraphics::setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size) {
    deferredLink.setPositionInWindow(this->offset = offset, this->size = size);
}

void DeferredGraphics::updateCommandBuffer(uint32_t imageIndex){
    for(auto& [name, workflow]: workflows){
        if (workflow->commandBuffer(imageIndex).dropFlag()) {
            workflow->beginCommandBuffer(imageIndex);
            workflow->updateCommandBuffer(imageIndex);
            workflow->endCommandBuffer(imageIndex);
        }
    }
}

void DeferredGraphics::updateBuffers(uint32_t imageIndex){
    CHECK(copyCommandBuffers[imageIndex].reset());
    CHECK(copyCommandBuffers[imageIndex].begin());

    if(cameraObject){
        cameraObject->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for(auto& object: objects){
        object->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for(auto& light: lights){
        light->update(imageIndex, copyCommandBuffers[imageIndex]);
    }

    CHECK(copyCommandBuffers[imageIndex].end());
}

void DeferredGraphics::createStorageBuffers(uint32_t imageCount){
    storageBuffersHost.resize(imageCount);
    for (auto& buffer : storageBuffersHost) {
        buffer.create(device->instance, device->getLogical(), sizeof(StorageBufferObject), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    bDatabase.buffersMap["storage"] = &storageBuffersHost;
}

void DeferredGraphics::updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey){
    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = moon::math::Vector<float,4>(mousex,mousey,0.0f,0.0f);
        StorageUBO.number = std::numeric_limits<uint32_t>::max();
        StorageUBO.depth = 1.0f;
    storageBuffersHost[currentImage].copy(&StorageUBO);
}

void DeferredGraphics::readStorageBuffer(uint32_t currentImage, uint32_t& primitiveNumber, float& depth){
    StorageBufferObject storageBuffer{};
    std::memcpy((void*)&storageBuffer, (void*)storageBuffersHost[currentImage].map(), sizeof(StorageBufferObject));
    primitiveNumber = storageBuffer.number;
    depth = storageBuffer.depth;
}

void DeferredGraphics::create(moon::interfaces::Model *pModel){
    pModel->create(*device, commandPool);
}

void DeferredGraphics::bind(moon::interfaces::Camera* cameraObject){
    this->cameraObject = cameraObject;
    cameraObject->create(*device, imageCount);
    bDatabase.addBufferData("camera", &cameraObject->getBuffers());
}

void DeferredGraphics::remove(moon::interfaces::Camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject = nullptr;
        bDatabase.buffersMap.erase("camera");
    }
}

void DeferredGraphics::bind(moon::interfaces::Light* lightSource){
    lightSource->create(*device, commandPool, imageCount);
    lights.push_back(lightSource);

    if (depthMaps.find(lightSource) == depthMaps.end()) {
        moon::utils::ImageInfo shadowsInfo{ imageCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, MSAASamples };
        depthMaps[lightSource] = utils::DepthMap(*device, commandPool, shadowsInfo);
        depthMaps[lightSource].update(lightSource->isShadowEnable() && enable["Shadow"]);
    }

    workflows["Shadow"]->raiseUpdateFlags();
    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Scattering"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < transparentLayersCount; i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };
}

bool DeferredGraphics::remove(moon::interfaces::Light* lightSource){
    size_t size = lights.size();
    lights.erase(std::remove(lights.begin(), lights.end(), lightSource), lights.end());

    if(depthMaps.count(lightSource)){
        depthMaps.erase(lightSource);
        workflows["Shadow"]->raiseUpdateFlags();
    }

    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Scattering"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < transparentLayersCount; i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };

    return size - objects.size() > 0;
}

void DeferredGraphics::bind(moon::interfaces::Object* object){
    object->create(*device, commandPool, imageCount);
    objects.push_back(object);

    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Skybox"]->raiseUpdateFlags();
    workflows["Shadow"]->raiseUpdateFlags();
    workflows["BoundingBox"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < transparentLayersCount; i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };
}

bool DeferredGraphics::remove(moon::interfaces::Object* object){
    size_t size = objects.size();
    objects.erase(std::remove(objects.begin(), objects.end(), object), objects.end());

    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Skybox"]->raiseUpdateFlags();
    workflows["Shadow"]->raiseUpdateFlags();
    workflows["BoundingBox"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < transparentLayersCount; i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };
    return size - objects.size() > 0;
}

DeferredGraphics& DeferredGraphics::setEnable(const std::string& name, bool enable){
    this->enable[name] = enable;
    if(workflows.count(name) > 0) workflows[name]->raiseUpdateFlags();
    return *this;
}

bool DeferredGraphics::getEnable(const std::string& name){
    return enable[name];
}

DeferredGraphics& DeferredGraphics::setMinAmbientFactor(const float& minAmbientFactor){
    static_cast<Graphics*>(workflows["DeferredGraphics"].get())->setMinAmbientFactor(minAmbientFactor);
    workflows["DeferredGraphics"]->raiseUpdateFlags();
    for(uint32_t i = 0; i < transparentLayersCount; i++){
        const auto key = "TransparentLayer" + std::to_string(i);
        static_cast<Graphics*>(workflows[key].get())->setMinAmbientFactor(minAmbientFactor);
        workflows[key]->raiseUpdateFlags();
    }

    return *this;
}

DeferredGraphics& DeferredGraphics::setScatteringRefraction(bool enable){
    static_cast<LayersCombiner*>(workflows["LayersCombiner"].get())->setScatteringRefraction(enable);
    workflows["LayersCombiner"]->raiseUpdateFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setBlitFactor(float blitFactor){
    if(enable["Bloom"] && blitFactor >= 1.0f){
        static_cast<moon::workflows::BloomGraphics*>(workflows["Bloom"].get())->setBlitFactor(blitFactor).setSamplerStepX(blitFactor).setSamplerStepY(blitFactor);
    }
    workflows["Bloom"]->raiseUpdateFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setBlurDepth(float blurDepth){
    static_cast<moon::workflows::GaussianBlur*>(workflows["Blur"].get())->setBlurDepth(blurDepth);
    static_cast<LayersCombiner*>(workflows["LayersCombiner"].get())->setBlurDepth(blurDepth);
    workflows["Blur"]->raiseUpdateFlags();
    workflows["LayersCombiner"]->raiseUpdateFlags();
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
