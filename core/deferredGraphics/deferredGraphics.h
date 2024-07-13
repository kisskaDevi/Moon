#ifndef DEFERREDGRAPHICS_H
#define DEFERREDGRAPHICS_H

#include "graphicsInterface.h"
#include "workflow.h"
#include "link.h"

#include "cursor.h"
#include "buffer.h"
#include "vector.h"

#include <unordered_map>
#include <filesystem>
#include <memory>

#include "graphics.h"
#include "layersCombiner.h"
#include "skybox.h"
#include "scattering.h"
#include "sslr.h"
#include "bloom.h"
#include "blur.h"
#include "boundingBox.h"
#include "ssao.h"
#include "selector.h"
#include "shadow.h"
#include "postProcessing.h"

namespace moon::interfaces {
class Model;
class Camera;
class Object;
class Light;
}

#include "camera.h"
#include "object.h"
#include "light.h"

namespace moon::utils {
class Texture;
class DepthMap;
struct Node;
}

namespace moon::deferredGraphics {

struct StorageBufferObject{
    alignas(16) moon::math::Vector<float,4> mousePosition;
    alignas(4)  uint32_t number;
    alignas(4)  float depth;
};

class DeferredGraphics: public moon::graphicsManager::GraphicsInterface{
private:
    std::filesystem::path               shadersPath;
    std::filesystem::path               workflowsShadersPath;
    VkExtent2D                          extent{0,0};
    VkSampleCountFlagBits               MSAASamples{VK_SAMPLE_COUNT_1_BIT};

    moon::utils::BuffersDatabase        bDatabase;
    moon::utils::AttachmentsDatabase    aDatabase;

    std::unordered_map<std::string, std::unique_ptr<moon::workflows::Workflow>> workflows;
    std::unordered_map<std::string, moon::workflows::Parameters*> workflowsParameters;
    Link deferredLink;

    moon::utils::Buffers storageBuffersHost;

    utils::vkDefault::CommandPool    commandPool;
    utils::vkDefault::CommandBuffers copyCommandBuffers;
    std::vector<moon::utils::Node>   nodes;

    uint32_t blitAttachmentsCount{8};
    uint32_t transparentLayersCount{2};

    moon::utils::Cursor* cursor{ nullptr };
    moon::interfaces::Camera* cameraObject{nullptr};
    interfaces::Objects objects;
    interfaces::Lights lights;
    interfaces::DepthMaps depthMaps;
    utils::TextureMap emptyTextures;

    void createGraphicsPasses();
    void createStages();

    GraphicsParameters graphicsParams;
    std::vector<GraphicsParameters> transparentLayersParams;
    LayersCombinerParameters layersCombinerParams;
    workflows::SkyboxParameters skyboxParams;
    workflows::ScatteringParameters scatteringParams;
    workflows::SSLRParameters SSLRParams;
    workflows::BloomParameters bloomParams;
    workflows::GaussianBlurParameters blurParams;
    workflows::BoundingBoxParameters bbParams;
    workflows::SSAOParameters SSAOParams;
    workflows::SelectorParameters selectorParams;
    workflows::PostProcessingParameters postProcessingParams;
    workflows::ShadowGraphicsParameters shadowGraphicsParameters;

    void update(uint32_t imageIndex) override;
    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

public:
    DeferredGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);

    void reset() override;
    void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size) override;

    bool getEnable(const std::string& name);
    DeferredGraphics& requestUpdate(const std::string& name);
    DeferredGraphics& setEnable(const std::string& name, bool enable);

    DeferredGraphics& setExtent(VkExtent2D extent);
    DeferredGraphics& setShadersPath(const std::filesystem::path& shadersPath);
    DeferredGraphics& setMinAmbientFactor(const float& minAmbientFactor);
    DeferredGraphics& setScatteringRefraction(bool enable);
    DeferredGraphics& setBlitFactor(float blitFactor);
    DeferredGraphics& setBlurDepth(float blurDepth);

    void create(moon::interfaces::Model* pModel);

    void bind(moon::interfaces::Camera* cameraObject);
    void remove(moon::interfaces::Camera* cameraObject);

    void bind(moon::interfaces::Object* object);
    bool remove(moon::interfaces::Object* object);

    void bind(moon::interfaces::Light* lightSource);
    bool remove(moon::interfaces::Light* lightSource);

    void bind(moon::utils::Cursor* cursor);
    bool remove(moon::utils::Cursor* cursor);
};

}
#endif // DEFERREDGRAPHICS_H
