#ifndef DEFERREDGRAPHICSINTERFACE_H
#define DEFERREDGRAPHICSINTERFACE_H

#include "../graphicsInterface.h"

#include "renderStages/graphics.h"
#include "renderStages/postProcessing.h"
#include "filters/blur.h"
#include "filters/customfilter.h"
#include "filters/sslr.h"
#include "filters/ssao.h"
#include "filters/layersCombiner.h"
#include "filters/skybox.h"

class deferredGraphicsInterface: public graphicsInterface
{
private:
    std::string                                 ExternalPath;
    uint32_t                                    imageCount;
    VkExtent2D                                  extent;
    VkSampleCountFlagBits                       MSAASamples;

    std::vector<deviceInfo>                     devicesInfo;

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

    deferredGraphics                            DeferredGraphics;
    gaussianBlur                                Blur;
    customFilter                                Filter;
    SSLRGraphics                                SSLR;
    SSAOGraphics                                SSAO;
    skyboxGraphics                              Skybox;
    layersCombiner                              LayersCombiner;
    postProcessingGraphics                      PostProcessing;
    std::vector<deferredGraphics>               TransparentLayers;
    uint32_t                                    TransparentLayersCount{2};

    bool                                        enableTransparentLayers{true};
    bool                                        enableSkybox{true};
    bool                                        enableBlur{true};
    bool                                        enableBloom{true};
    bool                                        enableSSLR{false};
    bool                                        enableSSAO{false};

    std::vector<VkBuffer>                       storageBuffers;
    std::vector<VkDeviceMemory>                 storageBuffersMemory;

    std::vector<VkCommandBuffer>                commandBuffers;
    std::vector<bool>                           updateCommandBufferFlags;
    std::vector<bool>                           updateShadowCommandBufferFlags;

    std::vector<VkCommandBuffer>                commandBufferSet;

    camera*                                     cameraObject{nullptr};
    texture*                                    emptyTexture{nullptr};

    void fastCreateFilterGraphics(filterGraphics* filter, uint32_t attachmentsNumber, attachments* attachments);
    void fastCreateGraphics(deferredGraphics* graphics, DeferredAttachments* attachments);
    void createStorageBuffers(uint32_t imageCount);
    void updateCommandBuffer(uint32_t imageIndex, VkCommandBuffer* commandBuffer);
public:
    deferredGraphicsInterface(const std::string& ExternalPath, VkExtent2D extent = {0,0}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);
    void destroyEmptyTextures();

    ~deferredGraphicsInterface();
    void destroyGraphics() override;

    void setDevicesInfo(uint32_t devicesInfoCount, deviceInfo* devicesInfo) override;
    void setSupportImageCount(VkSurfaceKHR* surface) override;
    void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface) override;
    void updateDescriptorSets() override;

    void createCommandBuffers() override;
    void updateCommandBuffers() override;
    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;
    void freeCommandBuffers() override;

    VkCommandBuffer*    getCommandBuffers(uint32_t& commandBuffersCount, uint32_t imageIndex) override;
    uint32_t            getImageCount() override;
    VkSwapchainKHR&     getSwapChain() override;

    void        updateCmdFlags();
    void        updateShadowCmdFlags();

    void        setExtent(VkExtent2D extent);
    void        setExternalPath(const std::string& ExternalPath);
    void        setEmptyTexture(std::string ZERO_TEXTURE);
    void        setMinAmbientFactor(const float& minAmbientFactor);

    void        createModel(gltfModel* pModel);
    void        destroyModel(gltfModel* pModel);

    void        bindCameraObject(camera* cameraObject);
    void        removeCameraObject(camera* cameraObject);

    void        bindSkyBoxObject(skyboxObject* object);
    bool        removeSkyBoxObject(skyboxObject* object);

    void        bindBaseObject(object* object);
    bool        removeObject(object* object);

    void        bindLightSource(light* lightSource);
    void        removeLightSource(light* lightSource);

    void        updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey);
    uint32_t    readStorageBuffer(uint32_t currentImage);
};

#endif // DEFERREDGRAPHICSINTERFACE_H
