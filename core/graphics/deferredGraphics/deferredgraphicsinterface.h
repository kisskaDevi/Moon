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

struct updateFlag{
    bool                                        enable{false};
    uint32_t                                    frames{0};
};

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
    std::vector<attachments>                    layersCombinedAttachment;

    deferredGraphics                            DeferredGraphics;
    gaussianBlur                                Blur;
    customFilter                                Filter;
    SSLRGraphics                                SSLR;
    SSAOGraphics                                SSAO;
    layersCombiner                              LayersCombiner;
    postProcessingGraphics                      PostProcessing;
    std::vector<deferredGraphics>               TransparentLayers;
    uint32_t                                    TransparentLayersCount{3};

    std::vector<VkBuffer>                       storageBuffers;
    std::vector<VkDeviceMemory>                 storageBuffersMemory;

    updateFlag                                  worldCmd;
    updateFlag                                  lightsCmd;
    updateFlag                                  worldUbo;
    updateFlag                                  lightsUbo;

    std::vector<VkCommandBuffer>                commandBuffers;
    std::vector<VkCommandBuffer>                commandBufferSet;

    void fastCreateFilterGraphics(filterGraphics* filter, uint32_t attachmentsNumber, attachments* attachments);
    void createStorageBuffers(uint32_t imageCount);
    void updateUniformBuffer(uint32_t imageIndex);
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
    void updateAllCommandBuffers() override;
    void updateCommandBuffers(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;
    void freeCommandBuffers() override;

    VkCommandBuffer*    getCommandBuffers(uint32_t& commandBuffersCount, uint32_t imageIndex) override;
    uint32_t            getImageCount() override;
    VkSwapchainKHR&     getSwapChain() override;

    void        resetCmdLight();
    void        resetCmdWorld();
    void        resetUboLight();
    void        resetUboWorld();

    void        setExtent(VkExtent2D extent);
    void        setExternalPath(const std::string& ExternalPath);
    void        setEmptyTexture(std::string ZERO_TEXTURE);
    void        setCameraObject(camera* cameraObject);

    void        createModel(gltfModel* pModel);
    void        destroyModel(gltfModel* pModel);

    void        bindLightSource(light* lightSource);
    void        removeLightSource(light* lightSource);

    void        bindBaseObject(object* newObject);
    void        bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH);

    bool        removeObject(object* object);
    bool        removeSkyBoxObject(object* object);

    void        setMinAmbientFactor(const float& minAmbientFactor);

    void        updateStorageBuffer(uint32_t currentImage, const float& mousex, const float& mousey);
    uint32_t    readStorageBuffer(uint32_t currentImage);
};

#endif // DEFERREDGRAPHICSINTERFACE_H
