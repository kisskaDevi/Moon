#include "graphicsManager.h"

#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <iostream>
#include <set>

graphicsManager::graphicsManager()
{}

//==========================Instance=============================================//

void graphicsManager::graphicsManager::createInstance()
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    auto extensions = getRequiredExtensions();

    //Структура описывает экземпляр Vulkan
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidationLayers && !checkValidationLayerSupport())
        throw std::runtime_error("validation layers requested, but not available!");

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers)
    {
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("failed to create instance!");
}
    std::vector<const char*> graphicsManager::getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if(enableValidationLayers)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }
    bool graphicsManager::checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers){
            bool layerFound = false;
            for(const auto& layerProperties: availableLayers){
                if(strcmp(layerName, layerProperties.layerName)==0){
                    layerFound = true;  break;
                }
            }
            if(!layerFound)
                return false;
        }

        return true;
    }

//===================================DebugMessenger====================================//

void graphicsManager::graphicsManager::setupDebugMessenger()
{
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug messenger!");
}
    void graphicsManager::graphicsManager::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }
        VKAPI_ATTR VkBool32 VKAPI_CALL graphicsManager::debugCallback(
         VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
         VkDebugUtilsMessageTypeFlagsEXT messageType,
         const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
         void* pUserData)
        {
            static_cast<void>(messageSeverity);
            static_cast<void>(messageType);
            static_cast<void>(pUserData);
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
        }
    VkResult graphicsManager::CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr)    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        else                    return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

//===========================Surface==========================//

void graphicsManager::createSurface(GLFWwindow* window)
{

    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface!");
}

//===========================Devices==========================//

void graphicsManager::graphicsManager::pickPhysicalDevice()
{
    /* После того как у нас есть экземпляр Vulkan, мы можем найти все совместимые с Vulkan.
     * В Vulkan есть два типа устройств - физические и логические. Физическое устройство -
     * это обычные части системы - графические карты, ускорители, DSP или другие компоненты.*/

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if(deviceCount==0)
        throw std::runtime_error("failed to find GPUs with Vulkan support!");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices)
    {
        std::vector<QueueFamilyIndices> indices = findQueueFamilies(device, surface);
        if (indices.size()!=0 && isDeviceSuitable(device,surface,deviceExtensions))
        {
            physicalDevice currentDevice = {device,indices};
            physicalDevices.push_back(currentDevice);
        }
    }

    if(physicalDevices.size()!=0)
    {
        physicalDeviceNumber = 1;
        indicesNumber = 0;
    }

    if (physicalDevices.at(physicalDeviceNumber).device == VK_NULL_HANDLE)
        throw std::runtime_error("failed to find a suitable GPU!");
}

void graphicsManager::createLogicalDevice()
{
    /* Логическое устройство - это программная абстракция физического устройсва, сконфигурированная в соответствии
     * с тем, как задано приложение. После выбора физического устройства вашему приложению необходимо создать
     * соответствующее ему логическое устройство.*/

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily.value(), physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).presentFamily.value()};  //массив очередей

    //Vulkan позволяет назначать приоритеты очередям, чтобы влиять на планирование выполнения командного буфера,
    //используя числа с плавающей запятой между 0.0 и 1.0. Это необходимо, даже если очередь одна:
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)        //по 2 очередям мы составляем следующие структуры
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;     //индекс семейства очередей из которого вы хотите выделять очереди
        queueCreateInfo.queueCount = 1;                     //количество очередей, которые вы выделяете
        queueCreateInfo.pQueuePriorities = &queuePriority;  //приоритет
        queueCreateInfos.push_back(queueCreateInfo);        //записываем в массив очередей
    }

    VkPhysicalDeviceFeatures deviceFeatures{};              //дополнительные возможности
        deviceFeatures.samplerAnisotropy = VK_TRUE;         //анизотропная фильтрация
        deviceFeatures.independentBlend = VK_TRUE;
        deviceFeatures.sampleRateShading = VK_TRUE;
        deviceFeatures.imageCubeArray = VK_TRUE;
        deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());   //количество очередей в массиве
        createInfo.pQueueCreateInfos = queueCreateInfos.data();                             //массив очередей соответственно
        createInfo.pEnabledFeatures = &deviceFeatures;                                      //поддерживаемые дополнительные возможности устройства
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());  //задаём количество расширений
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();                       //передём имена этих расширений
    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else
        createInfo.enabledLayerCount = 0;

    if (vkCreateDevice(physicalDevices.at(physicalDeviceNumber).device, &createInfo, nullptr, &device) != VK_SUCCESS)    //создание логического устройства
        throw std::runtime_error("failed to create logical device!");

    //Получение дескрипторов очереди из найденного семейства очередей от выбранного устройства
    vkGetDeviceQueue(device, physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).presentFamily.value(), 0, &presentQueue);
}

void graphicsManager::createCommandPool()
{
    /* Главной целью очереди является выполнение работы от имени вашего приложения.
     * Работа/задания представлены как последовательность команд, которые записываются
     * в командные буферы. Ваше приложение создаёт командные буферы, одержащие задания,
     * которые необходимо выполнить, и передает (submit) их в одну из очередей для выполения.
     * Прежде чем вы можете запоминать какие-либо  команды, вам нужно создать командный буфер.
     * Командные буферы не создаются явно, а выделяются из пулов.*/

    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily.value();              //задаёт семейство очередей, в которые будет передаваться созданные командные буферы
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;                                                                   //задаёт флаги, определяющие поведение пула и командных буферов, выделяемых из него
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)                                                        //создание пула команд
        throw std::runtime_error("failed to create command pool!");
}

void graphicsManager::createGraphics(graphicsInterface* graphics, GLFWwindow* window)
{
    this->graphics = graphics;

    std::vector<deviceInfo> info;
    info.push_back(deviceInfo{
                       &physicalDevices.at(physicalDeviceNumber).device,
                       &physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily,
                       &physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).presentFamily,
                       &device,
                       &graphicsQueue,
                       &commandPool
                   });

    graphics->createGraphics(window,&surface,static_cast<uint32_t>(info.size()),info.data());
}


void graphicsManager::createCommandBuffers()
{
    commandBuffers.resize(graphics->getImageCount());
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;                                 //дескриптор ранее созданного командного пула
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;                   //задаёт уровень командных буферов, которые вы хотите выделить
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();     //задаёт число командных буферов
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate command buffers!");

    graphics->updateCommandBuffers(commandBuffers.data());
}


void graphicsManager::createSyncObjects()
{
    /* Синхронизация в Vulkan реализуется при помощи использования различных примитивов синхронизации
     * Есть несколько типов примитивов синхронизации, и они предназначены для различных целей в вашем приложении
     * Тремя главными типами синхронизационых примитивов являются:
     * *барьеры (fence): используются, когда CPU необходимо дождаться, когда устройство завершит выполнение
     *  больших фрагментов работы, передаваемых в очереди, обычно при помощи операционной системы
     * *события: представляют точный примитив синхронизации, который может взводиться либо со стороны CPU, либо со стороны GPU.
     *  Он может взводиться в середине командного буфера, когда взводится устройством, и устройство может ожидать его в определённых местах конвейера
     * *семафоры: синхронизационные примитивы, которые используются для управления владением ресурсами между
     *  различными очередями одного и того же устройства. Они могут быть использованы для синхронизации работы, выполняемой
     *  на различных очередях, которая бы в противном случае выполнялась асинхронно*/

    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);      //изображение получено и готово к рендерингу
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);      //рендеренг завершён и может произойти презентация
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);                //синхронизация GPU и CPU
    imagesInFlight.resize(graphics->getImageCount(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t imageIndex = 0; imageIndex < MAX_FRAMES_IN_FLIGHT; imageIndex++)
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[imageIndex]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[imageIndex]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[imageIndex]) != VK_SUCCESS)
            throw std::runtime_error("failed to create synchronization objects for a frame!");
}

VkResult graphicsManager::checkNextFrame()
{
    VkResult result = vkAcquireNextImageKHR(device, graphics->getSwapChain(), UINT64_MAX , imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);    //Получить индекс следующего доступного презентабельного изображения

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        return result;

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)                                       //если нет следующего изображения
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);       //ждём
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    return result;
}

VkResult graphicsManager::drawFrame()
{
    graphics->updateCmd(imageIndex,commandBuffers.data());
    graphics->updateUbo(imageIndex);

    VkSemaphore                         waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags                waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore                         signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    std::vector<VkCommandBuffer>        commandbufferSet;

    graphics->fillCommandBufferSet(commandbufferSet,imageIndex);
    commandbufferSet.push_back(commandBuffers[imageIndex]);

    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = commandbufferSet.size();
        submitInfo.pCommandBuffers = commandbufferSet.data();
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;                                                    //восстановить симофоры в несингнальное положение
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)           //отправляет последовательность семафоров или командных буферов в очередь
        throw std::runtime_error("failed to submit draw command buffer!");

    VkPresentInfoKHR presentInfo{};
    VkSwapchainKHR swapChains[] = {graphics->getSwapChain()};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
    VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);                                                  //Поставить изображение в очередь для презентации

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    return result;
}

void graphicsManager::freeCommandBuffers()
{
    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()),commandBuffers.data());
    commandBuffers.clear();
}

void graphicsManager::graphicsManager::cleanup()
{
    freeCommandBuffers();

    for (size_t imageIndex = 0; imageIndex < MAX_FRAMES_IN_FLIGHT; imageIndex++)
    {
        vkDestroySemaphore(device, renderFinishedSemaphores[imageIndex], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[imageIndex], nullptr);
        vkDestroyFence(device, inFlightFences[imageIndex], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers)
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}
    void graphicsManager::DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
    {
        //VkDebugUtilsMessengerEXT Объект также должен быть очищен с помощью вызова vkDestroyDebugUtilsMessengerEXT
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if(func != nullptr)    func(instance, debugMessenger, pAllocator);
    }

uint32_t                            graphicsManager::getImageIndex(){return imageIndex;}
uint32_t                            graphicsManager::getCurrentFrame(){return currentFrame;}

void                                graphicsManager::deviceWaitIdle()
{
    vkDeviceWaitIdle(device);
}
