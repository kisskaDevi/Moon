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
#include "filters/shadow.h"
#include "core/device.h"

struct stage
{
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkQueue queue;
    VkFence fence;

    stage(  std::vector<VkCommandBuffer> commandBuffers,
            std::vector<VkPipelineStageFlags> waitStages,
            std::vector<VkSemaphore> waitSemaphores,
            std::vector<VkSemaphore> signalSemaphores,
            VkQueue queue,
            VkFence fence) :
        commandBuffers(commandBuffers),
        waitStages(waitStages),
        waitSemaphores(waitSemaphores),
        signalSemaphores(signalSemaphores),
        queue(queue),
        fence(fence)
    {}

    VkResult submit(){
        VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = waitSemaphores.size();
            submitInfo.pWaitSemaphores = waitSemaphores.data();
            submitInfo.pWaitDstStageMask = waitStages.data();
            submitInfo.commandBufferCount = commandBuffers.size();
            submitInfo.pCommandBuffers = commandBuffers.data();
            submitInfo.signalSemaphoreCount = signalSemaphores.size();
            submitInfo.pSignalSemaphores = signalSemaphores.data();
        return vkQueueSubmit(queue, 1, &submitInfo, fence);
    }
};

class deferredGraphicsInterface: public graphicsInterface
{
private:
    std::string                                 ExternalPath;
    uint32_t                                    imageCount;
    VkExtent2D                                  extent;
    VkSampleCountFlagBits                       MSAASamples;

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

    deferredGraphics                            DeferredGraphics;
    gaussianBlur                                Blur;
    customFilter                                Filter;
    SSLRGraphics                                SSLR;
    SSAOGraphics                                SSAO;
    skyboxGraphics                              Skybox;
    shadowGraphics                              Shadow;
    layersCombiner                              LayersCombiner;
    postProcessingGraphics                      PostProcessing;
    std::vector<deferredGraphics>               TransparentLayers;
    uint32_t                                    TransparentLayersCount{2};

    bool                                        enableTransparentLayers{true};
    bool                                        enableSkybox{true};
    bool                                        enableBlur{true};
    bool                                        enableBloom{true};
    bool                                        enableSSLR{true};
    bool                                        enableSSAO{true};

    std::vector<VkBuffer>                       storageBuffers;
    std::vector<VkDeviceMemory>                 storageBuffersMemory;

    VkCommandPool                               commandPool;
    std::vector<VkCommandBuffer>                commandBuffers;
    std::vector<std::vector<VkSemaphore>>       semaphores;

    std::vector<bool>                           updateCommandBufferFlags;

    camera*                                     cameraObject{nullptr};
    texture*                                    emptyTexture{nullptr};

    void fastCreateFilterGraphics(filterGraphics* filter, uint32_t attachmentsNumber, attachments* attachments);
    void fastCreateGraphics(deferredGraphics* graphics, DeferredAttachments* attachments);
    void createStorageBuffers(uint32_t imageCount);
public:
    deferredGraphicsInterface(const std::string& ExternalPath, VkExtent2D extent = {0,0}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);
    void destroyEmptyTextures();

    ~deferredGraphicsInterface();
    void destroyGraphics() override;
    void destroyCommandPool() override;

    void setDevices(uint32_t devicesCount, physicalDevice* devices) override;
    void setSupportImageCount(VkSurfaceKHR* surface) override;
    void createCommandPool() override;
    void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface) override;
    void updateDescriptorSets() override;

    void createCommandBuffers() override;
    void updateCommandBuffers() override;
    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;
    void freeCommandBuffers() override;

    uint32_t            getImageCount() override;
    VkSwapchainKHR&     getSwapChain() override;

    VkSemaphore sibmit(VkSemaphore externalSemaphore, VkFence& externalFence, uint32_t imageIndex) override;

    void        updateCmdFlags();

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
