#ifndef DEFERREDGRAPHICS_H
#define DEFERREDGRAPHICS_H

#include "graphicsInterface.h"
#include "link.h"
#include "workflow.h"

#include "buffer.h"
#include "vector.h"

#include <unordered_map>
#include <filesystem>

struct node;
class model;
class camera;
class object;
class light;
class workflow;
class depthMap;

struct StorageBufferObject{
    alignas(16) vector<float,4>    mousePosition;
    alignas(4)  uint32_t           number;
    alignas(4)  float              depth;
};

class deferredGraphics: public graphicsInterface{
private:
    std::filesystem::path                       shadersPath;
    std::filesystem::path                       workflowsShadersPath;
    VkExtent2D                                  extent{0,0};
    VkSampleCountFlagBits                       MSAASamples{VK_SAMPLE_COUNT_1_BIT};

    buffersDatabase bDatabase;
    attachmentsDatabase aDatabase;
    std::unordered_map<std::string, workflow*>  workflows;
    std::unordered_map<std::string, bool>       enable;
    class link                                  Link;

    buffers                                     storageBuffersHost;

    VkCommandPool                               commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer>                copyCommandBuffers;
    std::vector<bool>                           updateCommandBufferFlags;
    std::vector<node*>                          nodes;

    uint32_t                                    blitAttachmentsCount{8};
    uint32_t                                    TransparentLayersCount{2};

    camera*                                     cameraObject{nullptr};
    std::vector<object*>                        objects;
    std::vector<light*>                         lights;
    std::unordered_map<light*, depthMap*>       depthMaps;
    std::unordered_map<std::string, texture*>   emptyTextures;

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
    deferredGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);
    ~deferredGraphics();

    void destroy() override;
    void create() override;

    void update(uint32_t imageIndex) override;

    void setPositionInWindow(const vector<float,2>& offset, const vector<float,2>& size) override;

    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    void updateCmdFlags();

    bool getEnable(const std::string& name);
    deferredGraphics& setEnable(const std::string& name, bool enable);
    deferredGraphics& setExtent(VkExtent2D extent);
    deferredGraphics& setShadersPath(const std::filesystem::path& shadersPath);
    deferredGraphics& setMinAmbientFactor(const float& minAmbientFactor);
    deferredGraphics& setScatteringRefraction(bool enable);
    deferredGraphics& setBlitFactor(float blitFactor);
    deferredGraphics& setBlurDepth(float blurDepth);

    void create(model* pModel);
    void destroy(model* pModel);

    void bind(camera* cameraObject);
    void remove(camera* cameraObject);

    void bind(object* object);
    bool remove(object* object);

    void bind(light* lightSource);
    bool remove(light* lightSource);

    void updateStorageBuffer(uint32_t imageIndex, const float& mousex, const float& mousey);
    void readStorageBuffer(uint32_t imageIndex, uint32_t& primitiveNumber, float& depth);
};

#endif // DEFERREDGRAPHICS_H
