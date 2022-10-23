#ifndef DEFERREDGRAPHICSINTERFACE_H
#define DEFERREDGRAPHICSINTERFACE_H

#include "../graphicsInterface.h"

#include "graphics.h"
#include "postProcessing.h"
#include "customfilter.h"
#include "sslr.h"
#include "ssao.h"

struct updateFlag{
    bool                                        enable = false;
    uint32_t                                    frames = 0;
};

class deferredGraphicsInterface: public graphicsInterface
{
private:
    std::string                                 ExternalPath;
    uint32_t                                    imageCount;
    VkExtent2D                                  extent;
    VkSampleCountFlagBits                       MSAASamples;

    std::vector<deviceInfo>                     devicesInfo;

    VkSwapchainKHR                              swapChain;

    float                                       blitFactor = 1.5f;
    uint32_t                                    blitAttachmentCount = 8;
    std::vector<attachments>                    blitAttachments;
    attachments                                 blitAttachment;
    attachments                                 sslrAttachment;
    attachments                                 ssaoAttachment;

    deferredGraphics                            DeferredGraphics;
    customFilter                                Filter;
    SSLRGraphics                                SSLR;
    SSAOGraphics                                SSAO;
    postProcessing                              PostProcessing;
    std::vector<deferredGraphics>               TransparentLayers;
    uint32_t                                    TransparentLayersCount = 3;

    updateFlag                                  worldCmd;
    updateFlag                                  lightsCmd;
    updateFlag                                  worldUbo;
    updateFlag                                  lightsUbo;


    void updateUniformBuffer(uint32_t imageIndex);
    void updateCommandBuffer(uint32_t imageIndex, VkCommandBuffer* commandBuffer);
public:
    deferredGraphicsInterface(const std::string& ExternalPath, VkExtent2D extent = {0,0}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);
    ~deferredGraphicsInterface();
    void destroyGraphics() override;
    void destroyEmptyTextures();

    void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface, uint32_t devicesInfoCount, deviceInfo* devicesInfo) override;
    void updateDescriptorSets() override;
    void updateCommandBuffers(VkCommandBuffer* commandBuffers) override;
    void fillCommandBufferSet(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex) override;
    void updateCmd(uint32_t imageIndex, VkCommandBuffer* commandBuffers) override;
    void updateUbo(uint32_t imageIndex) override;

    uint32_t        getImageCount() override;
    VkSwapchainKHR& getSwapChain() override;

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

    void        addLightSource(spotLight* lightSource);
    void        removeLightSource(spotLight* lightSource);

    void        bindBaseObject(object* newObject);
    void        bindBloomObject(object* newObject);
    void        bindOneColorObject(object* newObject);
    void        bindStencilObject(object* newObject, float lineWidth, glm::vec4 lineColor);
    void        bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH);

    bool        removeBaseObject(object* object);
    bool        removeBloomObject(object* object);
    bool        removeOneColorObject(object* object);
    bool        removeStencilObject(object* object);
    bool        removeSkyBoxObject(object* object);

    void        removeBinds();

    void        setMinAmbientFactor(const float& minAmbientFactor);

    void        updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition);
    uint32_t    readStorageBuffer(uint32_t currentImage);
};

#endif // DEFERREDGRAPHICSINTERFACE_H
