#ifndef DEFERREDGRAPHICS_H
#define DEFERREDGRAPHICS_H

#include "graphicsInterface.h"
#include "workflow.h"

#include "buffer.h"
#include "vector.h"

#include <unordered_map>
#include <filesystem>

namespace moon::interfaces {
class Model;
class Camera;
class Object;
class Light;
}
namespace moon::utils {
class Texture;
class DepthMap;
struct Node;
}

namespace moon::deferredGraphics {

class Link;

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
    std::unordered_map<std::string, moon::workflows::Workflow*> workflows;
    std::unordered_map<std::string, bool> enable;
    Link* deferredLink;

    moon::utils::Buffers                        storageBuffersHost;

    VkCommandPool                               commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer>                copyCommandBuffers;
    std::vector<bool>                           updateCommandBufferFlags;
    std::vector<moon::utils::Node>              nodes;

    uint32_t                                    blitAttachmentsCount{8};
    uint32_t                                    TransparentLayersCount{2};

    moon::interfaces::Camera* cameraObject{nullptr};
    std::vector<moon::interfaces::Object*> objects;
    std::vector<moon::interfaces::Light*> lights;
    std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*> depthMaps;
    std::unordered_map<std::string, moon::utils::Texture*> emptyTextures;

    void createStorageBuffers(uint32_t imageCount);
    void createGraphicsPasses();
    void createCommandBuffers();
    void createCommandPool();
    void updateDescriptorSets();

    void freeCommandBuffers();
    void destroyCommandPool();
    void destroyEmptyTextures();

    void updateCommandBuffer(uint32_t imageIndex);
    void updateBuffers(uint32_t imageIndex);

public:
    DeferredGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);
    ~DeferredGraphics();

    void destroy() override;
    void create() override;

    void update(uint32_t imageIndex) override;

    void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size) override;

    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    void updateCmdFlags();

    bool getEnable(const std::string& name);
    DeferredGraphics& setEnable(const std::string& name, bool enable);
    DeferredGraphics& setExtent(VkExtent2D extent);
    DeferredGraphics& setShadersPath(const std::filesystem::path& shadersPath);
    DeferredGraphics& setMinAmbientFactor(const float& minAmbientFactor);
    DeferredGraphics& setScatteringRefraction(bool enable);
    DeferredGraphics& setBlitFactor(float blitFactor);
    DeferredGraphics& setBlurDepth(float blurDepth);

    void create(moon::interfaces::Model* pModel);
    void destroy(moon::interfaces::Model* pModel);

    void bind(moon::interfaces::Camera* cameraObject);
    void remove(moon::interfaces::Camera* cameraObject);

    void bind(moon::interfaces::Object* object);
    bool remove(moon::interfaces::Object* object);

    void bind(moon::interfaces::Light* lightSource);
    bool remove(moon::interfaces::Light* lightSource);

    void updateStorageBuffer(uint32_t imageIndex, const float& mousex, const float& mousey);
    void readStorageBuffer(uint32_t imageIndex, uint32_t& primitiveNumber, float& depth);
};

}
#endif // DEFERREDGRAPHICS_H
