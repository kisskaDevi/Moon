#ifndef VULKANCORE_H
#define VULKANCORE_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3native.h>

class camera;
struct gltfModel;

template <typename type> class light;
class spotLight;
template <> class light<spotLight>;

#include <iostream>
#include <vector>
#include <set>

#include "graphics/graphics.h"
#include "graphics/customfilter.h"

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 3;

const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";
const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct physicalDevice{
    VkPhysicalDevice                            device;
    std::vector<QueueFamilyIndices>             indices;
};

struct updateFlag{
    bool                                        enable = false;
    uint32_t                                    frames = 0;
};

class VkApplication
{
public:
    VkApplication();

    void initializeVulkan();
    void mainLoop(GLFWwindow* window, bool &framebufferResized);
    void cleanup();

    uint32_t                                    getImageCount();
    uint32_t                                    getCurrentFrame();

    void                                        resetCmdLight();
    void                                        resetCmdWorld();
    void                                        resetUboLight();
    void                                        resetUboWorld();

    void                                        setEmptyTexture(std::string ZERO_TEXTURE);
    void                                        setCameraObject(camera* cameraObject);

    void                                        createModel(gltfModel* pModel);
    void                                        destroyModel(gltfModel* pModel);

    void                                        addLightSource(light<spotLight>* lightSource);
    void                                        removeLightSource(light<spotLight>* lightSource);

    void                                        bindBaseObject(object* newObject);
    void                                        bindBloomObject(object* newObject);
    void                                        bindOneColorObject(object* newObject);
    void                                        bindStencilObject(object* newObject, float lineWidth, glm::vec4 lineColor);
    void                                        bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH);

    bool                                        removeBaseObject(object* object);
    bool                                        removeBloomObject(object* object);
    bool                                        removeOneColorObject(object* object);
    bool                                        removeStencilObject(object* object);
    bool                                        removeSkyBoxObject(object* object);

    void                                        removeBinds();

    void                                        setMinAmbientFactor(const float& minAmbientFactor);

    void                                        updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition);
    uint32_t                                    readStorageBuffer(uint32_t currentImage);

    void                                        deviceWaitIdle();

//step 1
    void                                        createInstance();
    void                                        setupDebugMessenger();
    void                                        createSurface(GLFWwindow* window);
//step 2
    void                                        pickPhysicalDevice();
    void                                        createLogicalDevice();
    void                                        checkSwapChainSupport();
//step 3
    void                                        createCommandPool();
//step 4
    void                                        createGraphics(GLFWwindow* window, VkExtent2D extent = {0,0}, VkSampleCountFlagBits MSAASamples = VK_SAMPLE_COUNT_1_BIT);
    void                                        updateDescriptorSets();
//step 5
    void                                        createCommandBuffers();
    void                                        createSyncObjects();
//step 5
    VkResult                                    drawFrame(float frametime, std::vector<object*> objects);
    void                                        destroyGraphics();
    void                                        freeCommandBuffers();

private:

    VkInstance                                  instance;
    VkSurfaceKHR                                surface;
    VkDebugUtilsMessengerEXT                    debugMessenger;

    std::vector<physicalDevice>                 physicalDevices;
    uint32_t                                    physicalDeviceNumber;
    uint32_t                                    indicesNumber;

    VkDevice                                    device;
    VkQueue                                     graphicsQueue;
    VkQueue                                     presentQueue;

    SwapChainSupportDetails                     swapChainSupport;
    uint32_t                                    imageCount;

    graphics                                    Graphics;
    customFilter                                Filter;
    postProcessing                              PostProcessing;

    VkCommandPool                               commandPool;
    std::vector<VkCommandBuffer>                commandBuffers;

    std::vector<VkSemaphore>                    imageAvailableSemaphores;
    std::vector<VkSemaphore>                    renderFinishedSemaphores;
    std::vector<VkFence>                        inFlightFences;
    std::vector<VkFence>                        imagesInFlight;

    uint32_t                                    currentFrame = 0;
    bool                                        framebufferResized = false;

    updateFlag                                  worldCmd;
    updateFlag                                  lightsCmd;
    updateFlag                                  worldUbo;
    updateFlag                                  lightsUbo;

    std::vector<light<spotLight>    *>          lightSources;

    //=================================InitializeVulkan===========================================//

        std::vector<const char*>                getRequiredExtensions();
        bool                                    checkValidationLayerSupport();

        void                                    populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
        static VKAPI_ATTR VkBool32 VKAPI_CALL   debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,VkDebugUtilsMessageTypeFlagsEXT messageType,const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,void* pUserData);
        VkResult                                CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);

        void                                    createCommandBuffer();
        void                                    updateCommandBuffer(uint32_t imageIndex);

    //=================================DrawLoop===========================================//
        void                                    updateCmd(uint32_t imageIndex);
        void                                    updateUbo(uint32_t imageIndex);
        void                                    updateUniformBuffer(uint32_t imageIndex);

    //=================================Cleanup===========================================//
        void                                    DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
};

#endif // VULKANCORE_H
