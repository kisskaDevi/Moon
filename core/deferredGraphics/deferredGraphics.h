#ifndef DEFERREDGRAPHICS_H
#define DEFERREDGRAPHICS_H

#include "graphicsInterface.h"
#include "workflow.h"
#include "link.h"

#include "cursor.h"
#include "buffer.h"
#include "vector.h"
#include "node.h"

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

#include "camera.h"
#include "object.h"
#include "light.h"
#include "model.h"

namespace moon::deferredGraphics {

class DeferredGraphics: public moon::graphicsManager::GraphicsInterface{
private:
    std::filesystem::path shadersPath;
    std::filesystem::path workflowsShadersPath;
    VkExtent2D            extent{0,0};
    VkSampleCountFlagBits MSAASamples{VK_SAMPLE_COUNT_1_BIT};

    moon::utils::BuffersDatabase     bDatabase;
    moon::utils::AttachmentsDatabase aDatabase;

    workflows::WorkflowsMap workflows;
    workflows::ParametersMap workflowsParameters;
    Link deferredLink;

    utils::vkDefault::CommandPool commandPool;
    utils::vkDefault::CommandBuffers copyCommandBuffers;
    utils::Nodes nodes;

    uint32_t blitAttachmentsCount{8};
    uint32_t transparentLayersCount{2};

    moon::utils::Cursor* cursor{ nullptr };
    moon::interfaces::Camera* cameraObject{nullptr};
    interfaces::Objects objects;
    interfaces::Lights lights;
    interfaces::DepthMaps depthMaps;
    utils::TextureMap emptyTextures;

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

    void createGraphicsPasses();
    void createStages();

    void update(uint32_t imageIndex) override;
    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

public:
    DeferredGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent);

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
