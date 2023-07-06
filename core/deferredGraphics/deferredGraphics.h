#ifndef DEFERREDGRAPHICS_H
#define DEFERREDGRAPHICS_H

#include "graphicsInterface.h"

#include "graphics.h"
#include "postProcessing.h"
#include "blur.h"
#include "customfilter.h"
#include "sslr.h"
#include "ssao.h"
#include "layersCombiner.h"
#include "skybox.h"
#include "shadow.h"
#include "device.h"
#include "buffer.h"

#include <glm.hpp>

#include <unordered_map>

class node;
class model;
class camera;

struct StorageBufferObject{
    alignas(16) glm::vec4           mousePosition;
    alignas(4)  int                 number;
    alignas(4)  float               depth;
};

struct frameScale
{
    float xScale{0.0f};
    float yScale{0.0f};
};

class deferredGraphics: public graphicsInterface
{
private:
    std::string                                 ExternalPath{};
    uint32_t                                    imageCount{0};
    frameScale                                  offsetScale;
    frameScale                                  extentScale;
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
    attachments                                 skyboxAttachment;
    std::vector<attachments>                    blitAttachments;
    std::vector<attachments>                    layersCombinedAttachment;
    float                                       blitFactor{1.5f};
    uint32_t                                    blitAttachmentCount{8};

    graphics                                    DeferredGraphics;
    gaussianBlur                                Blur;
    customFilter                                Filter;
    SSLRGraphics                                SSLR;
    SSAOGraphics                                SSAO;
    skyboxGraphics                              Skybox;
    shadowGraphics                              Shadow;
    layersCombiner                              LayersCombiner;
    postProcessingGraphics                      PostProcessing;
    std::vector<graphics>                       TransparentLayers;
    uint32_t                                    TransparentLayersCount{2};

    bool                                        enableTransparentLayers{true};
    bool                                        enableSkybox{true};
    bool                                        enableBlur{true};
    bool                                        enableBloom{true};
    bool                                        enableSSLR{true};
    bool                                        enableSSAO{true};

    std::vector<buffer>                         storageBuffersHost;

    VkCommandPool                               commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer>                copyCommandBuffers;
    std::vector<bool>                           updateCommandBufferFlags;
    std::vector<node*>                          nodes;

    camera*                                     cameraObject{nullptr};
    texture*                                    emptyTexture{nullptr};

    std::vector<VkCommandBuffer> getTransparentLayersCommandBuffers(uint32_t imageIndex);
    void createStorageBuffers(uint32_t imageCount);
public:
    deferredGraphics(const std::string& ExternalPath, frameScale offsetScale = {0.0f,0.0f}, frameScale extentScale = {1.0f,1.0f}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);

    ~deferredGraphics();
    void destroyGraphics() override;
    void destroyCommandPool() override;
    void freeCommandBuffers() override;
    void destroyEmptyTextures();

    void setSwapChain(swapChain* swapChainKHR) override;
    void setDevices(uint32_t devicesCount, physicalDevice* devices) override;
    void setImageCount(uint32_t imageCount) override;
    void createCommandPool() override;
    void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface) override;
    void updateDescriptorSets() override;

    void createCommandBuffers() override;
    void updateCommandBuffers() override;
    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;

    std::vector<std::vector<VkSemaphore>> sibmit(std::vector<std::vector<VkSemaphore>>& externalSemaphore, std::vector<VkFence>& externalFence, uint32_t imageIndex) override;

    void        updateCmdFlags();

    void        setOffset(frameScale offset);
    void        setExtent(frameScale extent);
    void        setFrameBufferExtent(VkExtent2D extent);
    void        setExternalPath(const std::string& ExternalPath);
    void        setEmptyTexture(std::string ZERO_TEXTURE);
    void        setMinAmbientFactor(const float& minAmbientFactor);

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
