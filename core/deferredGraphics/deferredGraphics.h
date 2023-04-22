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

class deferredGraphics: public graphicsInterface
{
private:
    std::string                                 ExternalPath{};
    uint32_t                                    imageCount{0};
    VkExtent2D                                  extent{0,0};
    VkSampleCountFlagBits                       MSAASamples{VK_SAMPLE_COUNT_1_BIT};

    std::vector<physicalDevice>                 devices;
    physicalDevice                              device;

    VkSwapchainKHR                              swapChain{VK_NULL_HANDLE};

    DeferredAttachments                         deferredAttachments;
    std::vector<DeferredAttachments>            transparentLayersAttachments;

    attachments                                 blurAttachment;
    float                                       blitFactor{1.5f};
    uint32_t                                    blitAttachmentCount{8};
    std::vector<attachments>                    blitAttachments;
    attachments                                 sslrAttachment;
    attachments                                 ssaoAttachment;
    attachments                                 skyboxAttachment;
    std::vector<attachments>                    layersCombinedAttachment;

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

    struct buffer{
        VkBuffer       instance{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        bool           updateFlag{true};
        void*          map{nullptr};
    };
    std::vector<buffer>                         storageBuffersHost;

    VkCommandPool                               commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer>                copyCommandBuffers;
    std::vector<bool>                           updateCommandBufferFlags;
    std::vector<node*>                          nodes;

    camera*                                     cameraObject{nullptr};
    texture*                                    emptyTexture{nullptr};

    void createStorageBuffers(uint32_t imageCount);
public:
    deferredGraphics(const std::string& ExternalPath, VkExtent2D extent = {0,0}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);

    ~deferredGraphics();
    void destroyGraphics() override;
    void destroyCommandPool() override;
    void freeCommandBuffers() override;
    void destroyEmptyTextures();

    void setDevices(uint32_t devicesCount, physicalDevice* devices) override;
    void setSupportImageCount(VkSurfaceKHR* surface) override;
    void createCommandPool() override;
    void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface) override;
    void updateDescriptorSets() override;

    void createCommandBuffers() override;
    void updateCommandBuffers() override;
    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;

    uint32_t getImageCount() override;
    VkSwapchainKHR& getSwapChain() override;

    std::vector<std::vector<VkSemaphore>> sibmit(std::vector<std::vector<VkSemaphore>>& externalSemaphore, std::vector<VkFence>& externalFence, uint32_t imageIndex) override;

    void        updateCmdFlags();

    void        setExtent(VkExtent2D extent);
    void        setExternalPath(const std::string& ExternalPath);
    void        setEmptyTexture(std::string ZERO_TEXTURE);
    void        setMinAmbientFactor(const float& minAmbientFactor);

    void        createModel(model* pModel);
    void        destroyModel(model* pModel);

    void        bindCameraObject(camera* cameraObject);
    void        removeCameraObject(camera* cameraObject);

    void        bindObject(object* object);
    bool        removeObject(object* object);

    void        bindLightSource(light* lightSource);
    void        removeLightSource(light* lightSource);

    void        updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey);
    uint32_t    readStorageBuffer(uint32_t currentImage);
};

#endif // DEFERREDGRAPHICS_H
