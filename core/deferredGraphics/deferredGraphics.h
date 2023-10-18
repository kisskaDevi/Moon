#ifndef DEFERREDGRAPHICS_H
#define DEFERREDGRAPHICS_H

#include "graphicsInterface.h"
#include "link.h"
#include "workflow.h"

#include "device.h"
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

struct StorageBufferObject{
    alignas(16) vector<float,4>    mousePosition;
    alignas(4)  int                number;
    alignas(4)  float              depth;
};

class deferredGraphics: public graphicsInterface{
private:
    std::filesystem::path                       shadersPath;
    uint32_t                                    imageCount{0};
    VkExtent2D                                  extent{0,0};
    VkOffset2D                                  offset{0,0};
    VkExtent2D                                  frameBufferExtent{0,0};
    VkSampleCountFlagBits                       MSAASamples{VK_SAMPLE_COUNT_1_BIT};

    std::vector<physicalDevice>                 devices;
    physicalDevice                              device;

    std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>> bufferMap;
    std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>> attachmentsMap;

    std::unordered_map<std::string, workflow*>  workflows;
    std::unordered_map<std::string, bool>       enable;
    link                                        Link;

    std::vector<buffer>                         storageBuffersHost;

    VkCommandPool                               commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer>                copyCommandBuffers;
    std::vector<bool>                           updateCommandBufferFlags;
    std::vector<node*>                          nodes;

    float                                       blitFactor{1.5f};
    uint32_t                                    blitAttachmentCount{8};
    uint32_t                                    TransparentLayersCount{2};

    swapChain*                                  swapChainKHR{nullptr};
    camera*                                     cameraObject{nullptr};
    std::vector<object*>                        objects;
    std::vector<light*>                         lights;
    std::unordered_map<std::string, texture*>   emptyTextures;

    void createStorageBuffers(uint32_t imageCount);
    void createGraphicsPasses();
    void createCommandBuffers();
    void createCommandPool();

    void freeCommandBuffers();
    void destroyCommandPool();
    void destroyEmptyTextures();
public:
    deferredGraphics(const std::filesystem::path& shadersPath, VkExtent2D extent, VkOffset2D offset = {0,0}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);
    ~deferredGraphics() = default;

    void destroyGraphics() override;

    void setDevices(uint32_t devicesCount, physicalDevice* devices) override;
    void setSwapChain(swapChain* swapChainKHR) override;

    void createGraphics() override;

    void updateDescriptorSets();
    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;

    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    linkable*   getLinkable() override;

    void        updateCmdFlags();

    void        setExtentAndOffset(VkExtent2D extent, VkOffset2D offset = {0,0});
    void        setShadersPath(const std::filesystem::path& shadersPath);
    void        setMinAmbientFactor(const float& minAmbientFactor);
    void        setScatteringRefraction(bool enable);

    void        create(model* pModel);
    void        destroy(model* pModel);

    void        bind(camera* cameraObject);
    void        remove(camera* cameraObject);

    void        bind(object* object);
    bool        remove(object* object);

    void        bind(light* lightSource);
    void        remove(light* lightSource);

    void        updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey);
    uint32_t    readStorageBuffer(uint32_t currentImage);

    deferredGraphics& setEnable(const std::string& name, bool enable);
};

#endif // DEFERREDGRAPHICS_H
