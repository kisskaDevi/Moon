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

#include "deferredLink.h"

#include <cstring>

namespace moon::deferredGraphics {

DeferredGraphics::DeferredGraphics(const Parameters& parameters):
    params(parameters)
{
    link = std::make_unique<Link>();

    transparentLayersParams.resize(params.transparentLayersCount());
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
}

void DeferredGraphics::reset()
{
    commandPool = utils::vkDefault::CommandPool(device->getLogical());
    copyCommandBuffers = commandPool.allocateCommandBuffers(resourceCount);

    aDatabase.destroy();
    emptyTextures["black"] = moon::utils::Texture::empty(*device, commandPool);
    emptyTextures["white"] = moon::utils::Texture::empty(*device, commandPool, false);
    aDatabase.addEmptyTexture("black", &emptyTextures["black"]);
    aDatabase.addEmptyTexture("white", &emptyTextures["white"]);

    createGraphicsPasses();
    createStages();
}

void DeferredGraphics::createGraphicsPasses(){
    CHECK_M(!commandPool,                       std::string("[ DeferredGraphics::createGraphicsPasses ] VkCommandPool is VK_NULL_HANDLE"));
    CHECK_M(device->instance == VK_NULL_HANDLE, std::string("[ DeferredGraphics::createGraphicsPasses ] VkPhysicalDevice is VK_NULL_HANDLE"));
    CHECK_M(params.cameraObject == nullptr,     std::string("[ DeferredGraphics::createGraphicsPasses ] camera is nullptr"));

    VkExtent2D extent{ params.extent[0], params.extent[1] };
    moon::utils::ImageInfo info{ resourceCount, swapChainKHR->info().Format, extent, params.MSAASamples };
    moon::utils::ImageInfo scatterInfo{ resourceCount, VK_FORMAT_R32G32B32A32_SFLOAT, extent, params.MSAASamples };
    moon::utils::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, params.MSAASamples };

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
    graphicsParams.minAmbientFactor = params.minAmbientFactor();
    graphicsParams.imageInfo = info;
    graphicsParams.shadersPath = params.shadersPath;

    skyboxParams.in.camera = graphicsParams.in.camera;
    skyboxParams.out.baseColor = "skybox.color";
    skyboxParams.out.bloom = "skybox.bloom";
    skyboxParams.imageInfo = info;
    skyboxParams.shadersPath = params.workflowsShadersPath;

    scatteringParams.in.camera = graphicsParams.in.camera;
    scatteringParams.in.depth = graphicsParams.out.depth;
    scatteringParams.out.scattering = "scattering";
    scatteringParams.imageInfo = scatterInfo;
    scatteringParams.shadersPath = params.workflowsShadersPath;

    SSLRParams.in.camera = graphicsParams.in.camera;
    SSLRParams.in.position = graphicsParams.out.position;
    SSLRParams.in.normal = graphicsParams.out.normal;
    SSLRParams.in.color = graphicsParams.out.image;
    SSLRParams.in.depth = graphicsParams.out.depth;
    SSLRParams.in.firstTransparency = graphicsParams.out.transparency + "0";
    SSLRParams.in.defaultDepthTexture = "white";
    SSLRParams.out.sslr = "sslr";
    SSLRParams.imageInfo = info;
    SSLRParams.shadersPath = params.workflowsShadersPath;

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
    layersCombinerParams.transparentLayersCount = workflowsParameters["TransparentLayer"]->enable ? params.transparentLayersCount() : 1;
    layersCombinerParams.enableScatteringRefraction = params.scatteringRefraction();
    layersCombinerParams.blurDepth = params.blurDepth();
    layersCombinerParams.imageInfo = info;
    layersCombinerParams.shadersPath = params.shadersPath;

    bloomParams.in.bloom = layersCombinerParams.out.bloom;
    bloomParams.out.bloom = "bloomFinal";
    bloomParams.blitAttachmentsCount = params.blitAttachmentsCount();
    bloomParams.blitFactor = params.blitFactor();
    bloomParams.xSamplerStep = params.blitFactor();
    bloomParams.ySamplerStep = params.blitFactor();
    bloomParams.imageInfo = info;
    bloomParams.shadersPath = params.workflowsShadersPath;

    blurParams.in.blur = layersCombinerParams.out.blur;
    blurParams.out.blur = "blured";
    blurParams.blurDepth = params.blurDepth();
    blurParams.imageInfo = info;
    blurParams.shadersPath = params.workflowsShadersPath;

    bbParams.in.camera = graphicsParams.in.camera;
    bbParams.out.boundingBox = "boundingBox";
    bbParams.imageInfo = info;
    bbParams.shadersPath = params.workflowsShadersPath;

    SSAOParams.in.camera = graphicsParams.in.camera;
    SSAOParams.in.position = graphicsParams.out.position;
    SSAOParams.in.normal = graphicsParams.out.normal;
    SSAOParams.in.color = graphicsParams.out.image;
    SSAOParams.in.depth = graphicsParams.out.depth;
    SSAOParams.in.defaultDepthTexture = "white";
    SSAOParams.out.ssao = "ssao";
    SSAOParams.imageInfo = info;
    SSAOParams.shadersPath = params.workflowsShadersPath;

    selectorParams.in.storageBuffer = "storage";
    selectorParams.in.position = graphicsParams.out.position;
    selectorParams.in.depth = graphicsParams.out.depth;
    selectorParams.in.transparency = graphicsParams.out.transparency;
    selectorParams.in.defaultDepthTexture = "white";
    selectorParams.out.selector = "selector";
    selectorParams.transparentLayersCount = workflowsParameters["TransparentLayer"]->enable ? params.transparentLayersCount() : 1;
    selectorParams.imageInfo = info;
    selectorParams.shadersPath = params.workflowsShadersPath;

    postProcessingParams.in.baseColor = layersCombinerParams.out.color;
    postProcessingParams.in.bloom = bloomParams.out.bloom;
    postProcessingParams.in.blur = blurParams.out.blur;
    postProcessingParams.in.boundingBox = bbParams.out.boundingBox;
    postProcessingParams.in.ssao = SSAOParams.out.ssao;
    postProcessingParams.out.postProcessing = "final";
    postProcessingParams.imageInfo = info;
    postProcessingParams.shadersPath = params.workflowsShadersPath,

    shadowGraphicsParameters.imageInfo = shadowsInfo;
    shadowGraphicsParameters.shadersPath = params.workflowsShadersPath,

    workflows.clear();

    workflows["DeferredGraphics"] = std::make_unique<Graphics>(graphicsParams, &objects, &lights, &depthMaps);
    workflows["LayersCombiner"] = std::make_unique<LayersCombiner>(layersCombinerParams);
    workflows["PostProcessing"] = std::make_unique<moon::workflows::PostProcessingGraphics>(postProcessingParams);

    for(uint32_t i = 0; i < params.transparentLayersCount(); i++){
        const auto key = "TransparentLayer" + std::to_string(i);
        transparentLayersParams[i].in = graphicsParams.in;
        transparentLayersParams[i].out = graphicsParams.out;
        transparentLayersParams[i].enable = workflowsParameters["TransparentLayer"]->enable;
        transparentLayersParams[i].enableTransparency = true;
        transparentLayersParams[i].transparencyPass = true;
        transparentLayersParams[i].transparencyNumber = i;
        transparentLayersParams[i].minAmbientFactor = params.minAmbientFactor();
        transparentLayersParams[i].imageInfo = graphicsParams.imageInfo;
        transparentLayersParams[i].shadersPath = graphicsParams.shadersPath;
        workflows[key] = std::make_unique<Graphics>(transparentLayersParams[i], &objects, &lights, &depthMaps);
    };

    workflows["Blur"] = std::make_unique<moon::workflows::GaussianBlur>(blurParams);
    workflows["Bloom"] = std::make_unique<moon::workflows::BloomGraphics>(bloomParams);
    workflows["Skybox"] = std::make_unique<moon::workflows::SkyboxGraphics>(skyboxParams, &objects);
    workflows["SSLR"] = std::make_unique<moon::workflows::SSLRGraphics>(SSLRParams);
    workflows["SSAO"] = std::make_unique<moon::workflows::SSAOGraphics>(SSAOParams);
    workflows["Shadow"] = std::make_unique<moon::workflows::ShadowGraphics>(shadowGraphicsParameters, &objects, &depthMaps);
    workflows["Scattering"] = std::make_unique<moon::workflows::Scattering>(scatteringParams, &lights, &depthMaps);
    workflows["BoundingBox"] = std::make_unique<moon::workflows::BoundingBoxGraphics>(bbParams, &objects);
    workflows["Selector"] = std::make_unique<moon::workflows::SelectorGraphics>(selectorParams, &params.cursor);

    for(auto& [_,workflow]: workflows){
        workflow->setDeviceProp(device->instance, device->getLogical());
        workflow->create(commandPool, aDatabase);
    }
    for (auto& [_, workflow] : workflows) {
        workflow->updateDescriptors(bDatabase, aDatabase);
    }

    moon::utils::ImageInfo linkInfo{resourceCount, swapChainKHR->info().Format, swapChainKHR->info().Extent, params.MSAASamples};
    link = std::make_unique<Link>(device->getLogical(), params.shadersPath, linkInfo, link->renderPass(), aDatabase.get(postProcessingParams.out.postProcessing));
}

void DeferredGraphics::createStages(){
    nodes.resize(resourceCount);
    for(uint32_t imageIndex = 0; imageIndex < resourceCount; imageIndex++){
        std::vector<VkCommandBuffer> transparentLayersCommandBuffers;
        for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
            transparentLayersCommandBuffers.push_back(workflows["TransparentLayer" + std::to_string(i)]->commandBuffer(imageIndex));
        }

        nodes[imageIndex] = moon::utils::PipelineNode(device->getLogical(), {
            moon::utils::PipelineStage(  {copyCommandBuffers[imageIndex]}, VK_PIPELINE_STAGE_TRANSFER_BIT, device->getQueue(0,0))
        }, new moon::utils::PipelineNode(device->getLogical(), {
            moon::utils::PipelineStage(  {workflows["Shadow"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0)),
            moon::utils::PipelineStage(  {workflows["Skybox"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->getQueue(0,0))
        }, new moon::utils::PipelineNode(device->getLogical(), {
            moon::utils::PipelineStage(  {workflows["DeferredGraphics"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0)),
            moon::utils::PipelineStage(  transparentLayersCommandBuffers, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::PipelineNode(device->getLogical(), {
            moon::utils::PipelineStage(  {workflows["Scattering"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0)),
            moon::utils::PipelineStage(  {workflows["SSLR"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::PipelineNode(device->getLogical(), {
            moon::utils::PipelineStage(  {workflows["LayersCombiner"]->commandBuffer(imageIndex)}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->getQueue(0,0))
        }, new moon::utils::PipelineNode(device->getLogical(), {
            moon::utils::PipelineStage(  {workflows["Selector"]->commandBuffer(imageIndex),
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

void DeferredGraphics::updateParameters() {
    if (params.blitFactor().updated()) {
        const auto blitFactor = params.blitFactor().release();
        bloomParams.blitFactor = blitFactor;
        bloomParams.xSamplerStep = blitFactor;
        bloomParams.ySamplerStep = blitFactor;
        workflows["Bloom"]->raiseUpdateFlags();
    }
    if (params.blurDepth().updated()) {
        const auto blurDepth = params.blurDepth().release();
        blurParams.blurDepth = blurDepth;
        layersCombinerParams.blurDepth = blurDepth;
        workflows["Blur"]->raiseUpdateFlags();
        workflows["LayersCombiner"]->raiseUpdateFlags();
    }
    if (params.minAmbientFactor().updated()) {
        const auto minAmbientFactor = params.minAmbientFactor().release();
        graphicsParams.minAmbientFactor = minAmbientFactor;
        workflows["DeferredGraphics"]->raiseUpdateFlags();
        for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
            transparentLayersParams[i].minAmbientFactor = minAmbientFactor;
            const auto key = "TransparentLayer" + std::to_string(i);
            workflows[key]->raiseUpdateFlags();
        };
    }
    if (params.scatteringRefraction().updated()) {
        layersCombinerParams.enableScatteringRefraction = params.scatteringRefraction().release();
        workflows["LayersCombiner"]->raiseUpdateFlags();
    }
}

void DeferredGraphics::update(uint32_t imageIndex) {
    updateParameters();

    CHECK(copyCommandBuffers[imageIndex].reset());
    CHECK(copyCommandBuffers[imageIndex].begin());
    if (params.cameraObject) {
        params.cameraObject->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for (auto& object : objects) {
        object->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for (auto& light : lights) {
        light->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    CHECK(copyCommandBuffers[imageIndex].end());

    for (auto& [name, workflow] : workflows) {
        workflow->update(imageIndex);
    }
}

void DeferredGraphics::create(moon::interfaces::Model *pModel){
    pModel->create(*device, commandPool);
}

void DeferredGraphics::bind(moon::interfaces::Camera* cameraObject){
    params.cameraObject = cameraObject;
    cameraObject->create(*device, resourceCount);
    bDatabase.add("camera", &cameraObject->getBuffers());
}

void DeferredGraphics::remove(moon::interfaces::Camera* cameraObject){
    if(params.cameraObject == cameraObject){
        params.cameraObject = nullptr;
        bDatabase.remove("camera");
    }
}

void DeferredGraphics::bind(moon::interfaces::Light* lightSource){
    lightSource->create(*device, commandPool, resourceCount);
    lights.push_back(lightSource);

    if (depthMaps.find(lightSource) == depthMaps.end()) {
        moon::utils::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, params.MSAASamples };
        depthMaps[lightSource] = utils::DepthMap(*device, commandPool, shadowsInfo);
        depthMaps[lightSource].update(lightSource->isShadowEnable() && workflowsParameters["Shadow"]->enable);
    }

    workflows["Shadow"]->raiseUpdateFlags();
    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Scattering"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
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
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
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
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
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
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };
    return size - objects.size() > 0;
}

void DeferredGraphics::bind(moon::utils::Cursor* cursor) {
    params.cursor = cursor;
    cursor->create(device->instance, device->getLogical());
    if (workflows["Selector"]) workflows["Selector"]->raiseUpdateFlags();
}

bool DeferredGraphics::remove(moon::utils::Cursor* cursor) {
    return params.cursor == cursor ? !(params.cursor = nullptr) : false;
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

Parameters& DeferredGraphics::parameters() {
    return params;
}

}
