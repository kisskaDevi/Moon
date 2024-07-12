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

#include <cstring>

namespace moon::deferredGraphics {

DeferredGraphics::DeferredGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples):
    shadersPath(shadersPath), workflowsShadersPath(workflowsShadersPath), extent(extent), MSAASamples(MSAASamples)
{
    transparentLayersParams.resize(transparentLayersCount);
    (workflowsParameters["DeferredGraphics"] = &graphicsParams)->enable = true;
    (workflowsParameters["LayersCombiner"] = &layersCombinerParams)->enable = true;
    (workflowsParameters["PostProcessing"] = &postProcessingParams)->enable = true;
    (workflowsParameters["Bloom"] = &bloomParams)->enable = false;
    (workflowsParameters["Blur"] = &blurParams)->enable = false;
    (workflowsParameters["Skybox"] = &skyboxParams)->enable = false;
    (workflowsParameters["SSLR"] = &SSLRParams)->enable = false;
    (workflowsParameters["SSAO"] = &SSAOParams)->enable = false;
    (workflowsParameters["Shadow"] = &shadowGraphicsParameters)->enable = false;
    (workflowsParameters["Scattering"] = &scatteringParams)->enable = false;
    (workflowsParameters["BoundingBox"] = &bbParams)->enable = false;
    (workflowsParameters["TransparentLayer"] = &transparentLayersParams.front())->enable = false;
    (workflowsParameters["Selector"] = &selectorParams)->enable = false;

    link = &deferredLink;
}

void DeferredGraphics::setPositionInWindow(const moon::math::Vector<float, 2>& offset, const moon::math::Vector<float, 2>& size) {
    deferredLink.setPositionInWindow(this->offset = offset, this->size = size);
}

void DeferredGraphics::reset()
{
    commandPool = utils::vkDefault::CommandPool(device->getLogical());

    aDatabase.destroy();
    emptyTextures["black"] = moon::utils::Texture::empty(*device, commandPool);
    emptyTextures["white"] = moon::utils::Texture::empty(*device, commandPool, false);
    aDatabase.addEmptyTexture("black", &emptyTextures["black"]);
    aDatabase.addEmptyTexture("white", &emptyTextures["white"]);

    createGraphicsPasses();
    createStages();
}

void DeferredGraphics::createGraphicsPasses(){
    CHECK_M(!commandPool,                        std::string("[ DeferredGraphics::createGraphicsPasses ] VkCommandPool is VK_NULL_HANDLE"));
    CHECK_M(device->instance == VK_NULL_HANDLE,  std::string("[ DeferredGraphics::createGraphicsPasses ] VkPhysicalDevice is VK_NULL_HANDLE"));
    CHECK_M(cameraObject == nullptr,             std::string("[ DeferredGraphics::createGraphicsPasses ] camera is nullptr"));

    graphicsParams.in.camera = "camera";
    graphicsParams.out.image = "image";
    graphicsParams.out.blur = "blur";
    graphicsParams.out.bloom = "bloom";
    graphicsParams.out.position = "GBuffer.position";
    graphicsParams.out.normal = "GBuffer.normal";
    graphicsParams.out.color = "GBuffer.color";
    graphicsParams.out.depth = "GBuffer.depth";
    graphicsParams.out.transparency = "transparency";
    graphicsParams.enableTransparency = workflowsParameters["TransparentLayer"]->enable;
    graphicsParams.transparencyPass = false;
    graphicsParams.transparencyNumber = 0;
    graphicsParams.minAmbientFactor = 0.05f;

    skyboxParams.in.camera = graphicsParams.in.camera;
    skyboxParams.out.baseColor = "skybox.color";
    skyboxParams.out.bloom = "skybox.bloom";

    scatteringParams.in.camera = graphicsParams.in.camera;
    scatteringParams.in.depth = graphicsParams.out.depth;
    scatteringParams.out.scattering = "scattering";

    SSLRParams.in.camera = graphicsParams.in.camera;
    SSLRParams.in.position = graphicsParams.out.position;
    SSLRParams.in.normal = graphicsParams.out.normal;
    SSLRParams.in.color = graphicsParams.out.image;
    SSLRParams.in.depth = graphicsParams.out.depth;
    SSLRParams.in.firstTransparency = graphicsParams.out.transparency + "0";
    SSLRParams.in.defaultDepthTexture = "white";
    SSLRParams.out.sslr = "sslr";

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
    layersCombinerParams.enableTransparentLayers = workflowsParameters["TransparentLayer"]->enable;
    layersCombinerParams.transparentLayersCount = workflowsParameters["TransparentLayer"]->enable ? transparentLayersCount : 1;

    bloomParams.in.bloom = layersCombinerParams.out.bloom;
    bloomParams.out.bloom = "bloomFinal";
    bloomParams.blitAttachmentsCount = blitAttachmentsCount;

    blurParams.in.blur = layersCombinerParams.out.blur;
    blurParams.out.blur = "blured";

    bbParams.in.camera = graphicsParams.in.camera;
    bbParams.out.boundingBox = "boundingBox";

    SSAOParams.in.camera = graphicsParams.in.camera;
    SSAOParams.in.position = graphicsParams.out.position;
    SSAOParams.in.normal = graphicsParams.out.normal;
    SSAOParams.in.color = graphicsParams.out.image;
    SSAOParams.in.depth = graphicsParams.out.depth;
    SSAOParams.in.defaultDepthTexture = "white";
    SSAOParams.out.ssao = "ssao";

    selectorParams.in.storageBuffer = "storage";
    selectorParams.in.position = graphicsParams.out.position;
    selectorParams.in.depth = graphicsParams.out.depth;
    selectorParams.in.transparency = graphicsParams.out.transparency;
    selectorParams.in.defaultDepthTexture = "white";
    selectorParams.out.selector = "selector";
    selectorParams.transparentLayersCount = workflowsParameters["TransparentLayer"]->enable ? transparentLayersCount : 1;

    postProcessingParams.in.baseColor = layersCombinerParams.out.color;
    postProcessingParams.in.bloom = bloomParams.out.bloom;
    postProcessingParams.in.blur = blurParams.out.blur;
    postProcessingParams.in.boundingBox = bbParams.out.boundingBox;
    postProcessingParams.in.ssao = SSAOParams.out.ssao;
    postProcessingParams.out.postProcessing = "final";

    moon::utils::ImageInfo info{ resourceCount, swapChainKHR->info().Format, extent, MSAASamples };
    moon::utils::ImageInfo scatterInfo{ resourceCount, VK_FORMAT_R32G32B32A32_SFLOAT, extent, MSAASamples };
    moon::utils::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, MSAASamples };

    workflows.clear();

    workflows["DeferredGraphics"] = std::make_unique<Graphics>(info, shadersPath, graphicsParams, &objects, &lights, &depthMaps);
    workflows["LayersCombiner"] = std::make_unique<LayersCombiner>(info, shadersPath, layersCombinerParams);
    workflows["PostProcessing"] = std::make_unique<moon::workflows::PostProcessingGraphics>(info, workflowsShadersPath, postProcessingParams);

    for(uint32_t i = 0; i < transparentLayersCount; i++){
        const auto key = "TransparentLayer" + std::to_string(i);
        transparentLayersParams[i].in = graphicsParams.in;
        transparentLayersParams[i].out = graphicsParams.out;
        transparentLayersParams[i].enable = workflowsParameters["TransparentLayer"]->enable;
        transparentLayersParams[i].enableTransparency = true;
        transparentLayersParams[i].transparencyPass = true;
        transparentLayersParams[i].transparencyNumber = i;
        transparentLayersParams[i].minAmbientFactor = 0.05f;
        workflows[key] = std::make_unique<Graphics>(info, shadersPath, transparentLayersParams[i], &objects, &lights, &depthMaps);
    };

    copyCommandBuffers = commandPool.allocateCommandBuffers(resourceCount);

    workflows["Blur"] = std::make_unique<moon::workflows::GaussianBlur>(info, workflowsShadersPath, blurParams);
    workflows["Bloom"] = std::make_unique<moon::workflows::BloomGraphics>(info, workflowsShadersPath, bloomParams);
    workflows["Skybox"] = std::make_unique<moon::workflows::SkyboxGraphics>(info, workflowsShadersPath, skyboxParams, &objects);
    workflows["SSLR"] = std::make_unique<moon::workflows::SSLRGraphics>(info, workflowsShadersPath, SSLRParams);
    workflows["SSAO"] = std::make_unique<moon::workflows::SSAOGraphics>(info, workflowsShadersPath, SSAOParams);
    workflows["Shadow"] = std::make_unique<moon::workflows::ShadowGraphics>(shadowsInfo, workflowsShadersPath, shadowGraphicsParameters, &objects, &depthMaps);
    workflows["Scattering"] = std::make_unique<moon::workflows::Scattering>(scatterInfo, workflowsShadersPath, scatteringParams, &lights, &depthMaps);
    workflows["BoundingBox"] = std::make_unique<moon::workflows::BoundingBoxGraphics>(info, workflowsShadersPath, bbParams, &objects);
    workflows["Selector"] = std::make_unique<moon::workflows::SelectorGraphics>(info, workflowsShadersPath, selectorParams, &cursor);

    for(auto& [_,workflow]: workflows){
        workflow->setDeviceProp(device->instance, device->getLogical());
        workflow->create(aDatabase);
    }
    for (auto& [_, workflow] : workflows) {
        workflow->updateDescriptorSets(bDatabase, aDatabase);
        workflow->createCommandBuffers(commandPool);
    }

    moon::utils::ImageInfo linkInfo{resourceCount, swapChainKHR->info().Format, swapChainKHR->info().Extent, MSAASamples};
    deferredLink.setShadersPath(shadersPath);
    deferredLink.setDeviceProp(device->getLogical());
    deferredLink.setImageCount(resourceCount);
    deferredLink.createDescriptorSetLayout();
    deferredLink.createPipeline(&linkInfo);
    deferredLink.createDescriptorPool();
    deferredLink.updateDescriptorSets(aDatabase.get("final"));
}

void DeferredGraphics::createStages(){
    std::vector<std::vector<VkCommandBuffer>> transparentLayersCommandBuffers(resourceCount);
    for (uint32_t imageIndex = 0; imageIndex < resourceCount; imageIndex++) {
        for (uint32_t i = 0; i < transparentLayersCount; i++) {
            transparentLayersCommandBuffers[imageIndex].push_back(workflows["TransparentLayer" + std::to_string(i)]->commandBuffer(imageIndex));
        }
    }

    nodes.resize(resourceCount);
    for(uint32_t imageIndex = 0; imageIndex < resourceCount; imageIndex++){
        nodes[imageIndex] = moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {copyCommandBuffers[imageIndex]}, VK_PIPELINE_STAGE_TRANSFER_BIT, device->getQueue(0,0))
        }, new moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {workflows["Shadow"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  {workflows["Skybox"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0))
        }, new moon::utils::Node(device->getLogical(), {
            moon::utils::Stage(  {workflows["DeferredGraphics"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0)),
            moon::utils::Stage(  transparentLayersCommandBuffers[imageIndex], VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
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

        CHECK(nodes[imageIndex].createSemaphores());
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
    CHECK(copyCommandBuffers[imageIndex].reset());
    CHECK(copyCommandBuffers[imageIndex].begin());
    if (cameraObject) {
        cameraObject->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for (auto& object : objects) {
        object->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for (auto& light : lights) {
        light->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    CHECK(copyCommandBuffers[imageIndex].end());

    for (auto& [name, workflow] : workflows) {
        if (workflow->commandBuffer(imageIndex).dropFlag()) {
            workflow->beginCommandBuffer(imageIndex);
            workflow->updateCommandBuffer(imageIndex);
            workflow->endCommandBuffer(imageIndex);
        }
    }
}

void DeferredGraphics::create(moon::interfaces::Model *pModel){
    pModel->create(*device, commandPool);
}

void DeferredGraphics::bind(moon::interfaces::Camera* cameraObject){
    this->cameraObject = cameraObject;
    cameraObject->create(*device, resourceCount);
    bDatabase.addBufferData("camera", &cameraObject->getBuffers());
}

void DeferredGraphics::remove(moon::interfaces::Camera* cameraObject){
    if(this->cameraObject == cameraObject){
        this->cameraObject = nullptr;
        bDatabase.buffersMap.erase("camera");
    }
}

void DeferredGraphics::bind(moon::interfaces::Light* lightSource){
    lightSource->create(*device, commandPool, resourceCount);
    lights.push_back(lightSource);

    if (depthMaps.find(lightSource) == depthMaps.end()) {
        moon::utils::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, MSAASamples };
        depthMaps[lightSource] = utils::DepthMap(*device, commandPool, shadowsInfo);
        depthMaps[lightSource].update(lightSource->isShadowEnable() && workflowsParameters["Shadow"]->enable);
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

    return size - lights.size() > 0;
}

void DeferredGraphics::bind(moon::interfaces::Object* object){
    object->create(*device, commandPool, resourceCount);
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

void DeferredGraphics::bind(moon::utils::Cursor* cursor) {
    this->cursor = cursor;
    cursor->create(device->instance, device->getLogical());
    if (workflows["Selector"]) workflows["Selector"]->raiseUpdateFlags();
}

bool DeferredGraphics::remove(moon::utils::Cursor* cursor) {
    return this->cursor == cursor ? !(this->cursor = nullptr) : false;
}

DeferredGraphics& DeferredGraphics::requestUpdate(const std::string& name) {
    workflows[name]->raiseUpdateFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setEnable(const std::string& name, bool enable){
    workflowsParameters[name]->enable = enable;
    return *this;
}

bool DeferredGraphics::getEnable(const std::string& name){
    return workflowsParameters[name]->enable;
}

DeferredGraphics& DeferredGraphics::setMinAmbientFactor(const float& minAmbientFactor){
    graphicsParams.minAmbientFactor = minAmbientFactor;
    workflows["DeferredGraphics"]->raiseUpdateFlags();
    for(uint32_t i = 0; i < transparentLayersCount; i++){
        transparentLayersParams[i].minAmbientFactor = minAmbientFactor;
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    }
    return *this;
}

DeferredGraphics& DeferredGraphics::setScatteringRefraction(bool enable){
    layersCombinerParams.enableScatteringRefraction = enable;
    workflows["LayersCombiner"]->raiseUpdateFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setBlitFactor(float blitFactor){
    if(workflowsParameters["Bloom"]->enable && blitFactor >= 1.0f){
        bloomParams.blitFactor = blitFactor;
        bloomParams.xSamplerStep = blitFactor;
        bloomParams.ySamplerStep = blitFactor;
    }
    workflows["Bloom"]->raiseUpdateFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setBlurDepth(float blurDepth){
    layersCombinerParams.blurDepth = blurDepth;
    blurParams.blurDepth = blurDepth;
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
