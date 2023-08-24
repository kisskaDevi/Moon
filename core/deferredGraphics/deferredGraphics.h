#ifndef DEFERREDGRAPHICS_H
#define DEFERREDGRAPHICS_H

#include "graphicsInterface.h"

#include "graphics.h"
#include "postProcessing.h"
#include "link.h"
#include "blur.h"
#include "customFilter.h"
#include "sslr.h"
#include "ssao.h"
#include "layersCombiner.h"
#include "scattering.h"
#include "skybox.h"
#include "shadow.h"
#include "device.h"
#include "buffer.h"

#include "vector.h"

#include <unordered_map>
#include <filesystem>

struct node;
class model;
class camera;

struct StorageBufferObject{
    alignas(16) vector<float,4>    mousePosition;
    alignas(4)  int                number;
    alignas(4)  float              depth;
};

class deferredGraphics: public graphicsInterface
{
private:
    std::filesystem::path                       shadersPath;
    uint32_t                                    imageCount{0};
    VkExtent2D                                  extent{0,0};
    VkOffset2D                                  offset{0,0};
    VkExtent2D                                  frameBufferExtent{0,0};
    VkSampleCountFlagBits                       MSAASamples{VK_SAMPLE_COUNT_1_BIT};

    std::vector<physicalDevice>                 devices;
    physicalDevice                              device;

    swapChain*                                  swapChainKHR{nullptr};

    DeferredAttachments                         deferredAttachments;
    std::vector<DeferredAttachments>            transparentLayersAttachments;

    attachments                                 blurAttachment;
    attachments                                 sslrAttachment;
    attachments                                 ssaoAttachment;
    attachments                                 scatteringAttachment;
    std::vector<attachments>                    skyboxAttachment;
    std::vector<attachments>                    blitAttachments;
    std::vector<attachments>                    layersCombinedAttachment;
    attachments                                 finalAttachment;

    graphics                                    DeferredGraphics;
    gaussianBlur                                Blur;
    customFilter                                Filter;
    SSLRGraphics                                SSLR;
    SSAOGraphics                                SSAO;
    skyboxGraphics                              Skybox;
    shadowGraphics                              Shadow;
    layersCombiner                              LayersCombiner;
    scattering                                  Scattering;
    postProcessingGraphics                      PostProcessing;
    link                                        Link;
    std::vector<graphics>                       TransparentLayers;
    bool                                        enableTransparentLayers{true};
    bool                                        enableSkybox{true};
    bool                                        enableBlur{true};
    bool                                        enableBloom{true};
    bool                                        enableSSLR{true};
    bool                                        enableSSAO{true};
    bool                                        enableScattering{true};

    std::vector<buffer>                         storageBuffersHost;

    VkCommandPool                               commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer>                copyCommandBuffers;
    std::vector<bool>                           updateCommandBufferFlags;
    std::vector<node*>                          nodes;

    float                                       blitFactor{1.5f};
    uint32_t                                    blitAttachmentCount{8};
    uint32_t                                    TransparentLayersCount{2};

    camera*                                     cameraObject{nullptr};
    texture*                                    emptyTextureBlack{nullptr};
    texture*                                    emptyTextureWhite{nullptr};

    std::vector<VkCommandBuffer> getTransparentLayersCommandBuffers(uint32_t imageIndex);
    void createStorageBuffers(uint32_t imageCount);
public:
    deferredGraphics(const std::filesystem::path& shadersPath, VkExtent2D extent, VkOffset2D offset = {0,0}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);

    ~deferredGraphics();
    void destroyGraphics() override;
    void destroyCommandPool();
    void freeCommandBuffers();
    void destroyEmptyTextures();

    void setSwapChain(swapChain* swapChainKHR) override;
    void setDevices(uint32_t devicesCount, physicalDevice* devices) override;
    void createCommandPool();
    void createGraphics(GLFWwindow* window, VkSurfaceKHR surface) override;

    void createGraphicsPasses(GLFWwindow* window, VkSurfaceKHR surface);
    void updateDescriptorSets();
    void createCommandBuffers();
    void updateCommandBuffers();
    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;

    std::vector<std::vector<VkSemaphore>> sibmit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex) override;

    linkable* getLinkable() override;

    void        updateCmdFlags();

    void        setExtentAndOffset(VkExtent2D extent, VkOffset2D offset = {0,0});
    void        setFrameBufferExtent(VkExtent2D extent);
    void        setShadersPath(const std::filesystem::path& shadersPath);
    void        setMinAmbientFactor(const float& minAmbientFactor);
    void        setScatteringRefraction(bool enable);

    void        createEmptyTexture();

    void        createModel(model* pModel);
    void        destroyModel(model* pModel);

    void        bindCameraObject(camera* cameraObject, bool create = false);
    void        removeCameraObject(camera* cameraObject);

    void        bindObject(object* object, bool create = false);
    bool        removeObject(object* object);

    void        bindLightSource(light* lightSource, bool create = false);
    void        removeLightSource(light* lightSource);

    void        updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey);
    uint32_t    readStorageBuffer(uint32_t currentImage);
};

#endif // DEFERREDGRAPHICS_H
