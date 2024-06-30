#ifndef LAYERSCOMBINER_H
#define LAYERSCOMBINER_H

#include "workflow.h"

namespace moon::deferredGraphics {

struct LayersCombinerAttachments{
    moon::utils::Attachments color;
    moon::utils::Attachments bloom;
    moon::utils::Attachments blur;

    static inline uint32_t size() { return 3;}
    inline moon::utils::Attachments* operator&(){ return &color;}
};

struct LayersCombinerParameters{
    struct{
        std::string camera;
        std::string color;
        std::string bloom;
        std::string position;
        std::string normal;
        std::string depth;
        std::string skyboxColor;
        std::string skyboxBloom;
        std::string scattering;
        std::string sslr;
        std::string transparency;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string color;
        std::string bloom;
        std::string blur;
    }out;
};

class LayersCombiner : public moon::workflows::Workflow
{
private:
    LayersCombinerParameters parameters;
    LayersCombinerAttachments frame;
    bool enable{true};
    float blurDepth{1.0f};

    struct Combiner : public moon::workflows::Workbody {
        uint32_t transparentLayersCount{ 1 };
        bool enableTransparentLayers{ true };
        bool enableScatteringRefraction{ true };

        Combiner(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
    }combiner;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    LayersCombiner(
        const moon::utils::ImageInfo& imageInfo,
        const std::filesystem::path& shadersPath,
        LayersCombinerParameters parameters,
        bool enable,
        uint32_t transparentLayersCount,
        bool enableScatteringRefraction);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    void setTransparentLayersCount(uint32_t transparentLayersCount);
    void setScatteringRefraction(bool enable);
    void setBlurDepth(float blurDepth);
};

}
#endif // LAYERSCOMBINER_H
