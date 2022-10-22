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

    uint32_t                                    devicesInfoCount;
    std::vector<deviceInfo>                     devicesInfo;

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
    deferredGraphicsInterface(const std::string& ExternalPath);
    ~deferredGraphicsInterface();
    void destroyGraphics() override;
    void destroyEmptyTextures();

    void createGraphics(uint32_t& imageCount, GLFWwindow* window, VkSurfaceKHR surface, VkExtent2D extent, VkSampleCountFlagBits MSAASamples, uint32_t devicesInfoCount, deviceInfo* devicesInfo) override;
    void updateDescriptorSets() override;
    void updateCommandBuffers(uint32_t imageCount, VkCommandBuffer* commandBuffers) override;
    void fillCommandbufferSet(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex) override;
    void updateCmd(uint32_t imageIndex, VkCommandBuffer* commandBuffers) override;
    void updateUbo(uint32_t imageIndex) override;

    VkSwapchainKHR& getSwapChain() override;

    void        resetCmdLight();
    void        resetCmdWorld();
    void        resetUboLight();
    void        resetUboWorld();

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
