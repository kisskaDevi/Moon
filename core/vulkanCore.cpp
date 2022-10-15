#include "vulkanCore.h"
#include "transformational/camera.h"
#include "transformational/light.h"
#include "transformational/object.h"
#include "transformational/gltfmodel.h"

#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <iostream>
#include <set>

VkApplication::VkApplication()
{}

//==========================Instance=============================================//

void VkApplication::VkApplication::createInstance()
{
    /*Экземпляр Vulkan - это программная конструкция, которая которая логически отделяет
     * состояние вашего приложения от других приложений или от библиотек, выполняемых в
     * контексте вашего приложения.*/

    //Структура короая описывает ваше приложение, исползуется при создании экземплряра VkInstanceCreateInfo
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";            //имя приложения
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);  //это версия вашего приложения
    appInfo.pEngineName = "No Engine";                      //название движка или библиотеки вашего приложения
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);       //версия движка или библиотеки вашего приложения
    appInfo.apiVersion = VK_API_VERSION_1_0;                //содержит версию вулкана на которое расчитано ваше приложение

    auto extensions = getRequiredExtensions();                                      //получить расширения, см далее в VkInstanceCreateInfo

    //Структура описывает экземпляр Vulkan
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;                                         //cтруктура описание которой приведена выше
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());    //число расширений которое вы хотите включить
    createInfo.ppEnabledExtensionNames = extensions.data();                         //и их имена

    if (enableValidationLayers && !checkValidationLayerSupport())
        throw std::runtime_error("validation layers requested, but not available!");

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) //если включены уравни проверки
    {
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());  //число слоёв экземпляра, которое вы хотите разрешить
        createInfo.ppEnabledLayerNames = validationLayers.data();                       //и их имена соответственно
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;      //поле позволяет передать функции связанный список стуктур
    } else {
        createInfo.enabledLayerCount = 0;                                               //число слоёв экземпляра
        createInfo.pNext = nullptr;                                                     //поле позволяет передать функции связанный список стуктур
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)                //создаём экземпляр функцией vkCreateInstance, в случае успеха возвращает VK_SUCCESS
        throw std::runtime_error("failed to create instance!");                         //параметр pAllocator указывает на аллокатор памяти CPU который ваше приложение может передать для управления используемой Vulkan памятью
}
    std::vector<const char*> VkApplication::getRequiredExtensions()                     //создадим getRequiredExtensions функцию, которая будет возвращать требуемый список расширений в зависимости от того, включены ли уровни проверки или нет
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if(enableValidationLayers)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); //макрос, который равен буквальной строке «VK_EXT_debug_utils»

        return extensions;
    }
    bool VkApplication::checkValidationLayerSupport()
    {
        uint32_t layerCount;                                        //количество слоёв экземпляра
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);   //если pProperties равен nullptr, то pPropertyCount должно указывать на переменную, в которую будет записано число доступных Vulkan слоёв

        std::vector<VkLayerProperties> availableLayers(layerCount);                 //массив структур в которые будет записана информация о зарегистрированных слоях проверки
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());    //на этом моменте будет произведена запись

        for (const char* layerName : validationLayers)                  //берём из перемененной validationLayers строки
        {
            bool layerFound = false;

            for(const auto& layerProperties: availableLayers)           //берём из локальной переменной availableLayers
            {
                if(strcmp(layerName, layerProperties.layerName)==0)     //и сравниваем, если хотя бы одно совпадение есть, выходим из цикла и попадаем в конец внешнего цикла
                {
                    layerFound = true;
                    break;
                }
            }

            if(!layerFound)                                             //если не нашли, возвращаем false
                return false;
        }

        return true;
    }

//===================================DebugMessenger====================================//

void VkApplication::VkApplication::setupDebugMessenger()
{
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug messenger!");
}
    void VkApplication::VkApplication::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }
        VKAPI_ATTR VkBool32 VKAPI_CALL VkApplication::debugCallback(
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
    VkResult VkApplication::CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr)    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        else                    return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

//===========================Surface==========================//

void VkApplication::createSurface(GLFWwindow* window)
{
    /*Объект, в который осуществляется рендеринг для показа пользователю, называется поверхностью(surface) и представлен при помощи дескриптора VkSurfaceKHR.
     * Это специальный объект, вводимый расширением VK_KHR_surface. Это расширение вводит общую функциональнать для работы с поверхностями, которая в дальнейшем
     * адаптируется под каждую платформу лоя получения платформенно-зависимого интерфейса для связи с поверхостью окна*/

    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)         //Эта функция создает поверхность Vulkan для указанного окна window.
        throw std::runtime_error("failed to create window surface!");
}

//===========================Devices==========================//

void VkApplication::VkApplication::pickPhysicalDevice()
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

void VkApplication::createLogicalDevice()
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

void VkApplication::createCommandPool()
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

void VkApplication::createGraphics(GLFWwindow* window, VkExtent2D extent, VkSampleCountFlagBits MSAASamples)
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevices.at(physicalDeviceNumber).device,surface);                          //здест происходит запрос поддерживаемы режимов и форматов которые в следующий строчках передаются в соответствующие переменные через фукцнии
    imageCount = swapChainSupport.capabilities.minImageCount + 1;                                                               //запрос на поддержк уминимального количества числа изображений, число изображений равное 2 означает что один буфер передний, а второй задний
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)            //в первом условии мы проверяем доступно ли нам вообще какое-то количество изображений и проверяем не совпадает ли максимальное число изображений с минимальным
        imageCount = swapChainSupport.capabilities.maxImageCount;

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);

    if(extent.height==0&&extent.width==0)
        extent = chooseSwapExtent(window, swapChainSupport.capabilities);

    if(MSAASamples != VK_SAMPLE_COUNT_1_BIT){
        VkSampleCountFlagBits maxMSAASamples = getMaxUsableSampleCount(physicalDevices.at(physicalDeviceNumber).device);
        if(MSAASamples>maxMSAASamples)  MSAASamples = maxMSAASamples;
    }

    DeferredGraphics.setExternalPath(ExternalPath);
    PostProcessing.setExternalPath(ExternalPath);
    Filter.setExternalPath(ExternalPath);
    SSLR.setExternalPath(ExternalPath);
    SSAO.setExternalPath(ExternalPath);

    DeferredGraphics.setDeviceProp(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool);
    PostProcessing.setDeviceProp(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool,&physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber),&surface);
    Filter.setDeviceProp(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool);
    SSLR.setDeviceProp(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool);
    SSAO.setDeviceProp(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool);

    imageInfo info{};
        info.Count = imageCount;
        info.Format = surfaceFormat.format;
        info.Extent = extent;
        info.Samples = MSAASamples;
    PostProcessing.setImageProp(&info);
    DeferredGraphics.setImageProp(&info);
    Filter.setImageProp(&info);
    SSLR.setImageProp(&info);
    SSAO.setImageProp(&info);

    DeferredGraphics.createAttachments();
    DeferredGraphics.createRenderPass();
    DeferredGraphics.createFramebuffers();
    DeferredGraphics.createPipelines();
    DeferredGraphics.createStorageBuffers(imageCount);
    DeferredGraphics.createBaseDescriptorPool();
    DeferredGraphics.createBaseDescriptorSets();
    DeferredGraphics.createSkyboxDescriptorPool();
    DeferredGraphics.createSkyboxDescriptorSets();
    DeferredGraphics.createSpotLightingDescriptorPool();
    DeferredGraphics.createSpotLightingDescriptorSets();

    PostProcessing.createAttachments(window, swapChainSupport);
    PostProcessing.createRenderPass();
    PostProcessing.createFramebuffers();
    PostProcessing.createPipelines();
    PostProcessing.createDescriptorPool();
    PostProcessing.createDescriptorSets(DeferredGraphics.getDeferredAttachments());

    Filter.setBlitAttachments(&PostProcessing.getBlitAttachment());
    Filter.setAttachments(PostProcessing.getBlitAttachments().size(),PostProcessing.getBlitAttachments().data());
    Filter.createRenderPass();
    Filter.createFramebuffers();
    Filter.createPipelines();
    Filter.createDescriptorPool();
    Filter.createDescriptorSets();
    Filter.updateSecondDescriptorSets();

    SSLR.setSSLRAttachments(&PostProcessing.getSSLRAttachment());
    SSLR.createRenderPass();
    SSLR.createFramebuffers();
    SSLR.createPipelines();
    SSLR.createDescriptorPool();
    SSLR.createDescriptorSets();
    SSLR.updateSecondDescriptorSets(DeferredGraphics.getDeferredAttachments(),DeferredGraphics.getSceneBuffer().data());

    SSAO.setSSAOAttachments(&PostProcessing.getSSAOAttachment());
    SSAO.createRenderPass();
    SSAO.createFramebuffers();
    SSAO.createPipelines();
    SSAO.createDescriptorPool();
    SSAO.createDescriptorSets();
    SSAO.updateSecondDescriptorSets(DeferredGraphics.getDeferredAttachments(),DeferredGraphics.getSceneBuffer().data());

//    TransparentLayers.resize(TransparentLayersCount);
//    for(uint32_t i=0;i<TransparentLayersCount;i++){
//        TransparentLayers[i].setDeviceProp(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool);
//        TransparentLayers[i].setImageProp(&info);
//        TransparentLayers[i].setTransparencyPass(true);
//        TransparentLayers[i].createAttachments();
//        TransparentLayers[i].createRenderPass();
//        TransparentLayers[i].createFramebuffers();
//        TransparentLayers[i].createPipelines();
//        TransparentLayers[i].createStorageBuffers(imageCount);
//        TransparentLayers[i].createBaseDescriptorPool();
//        TransparentLayers[i].createBaseDescriptorSets();
//        TransparentLayers[i].createSkyboxDescriptorPool();
//        TransparentLayers[i].createSkyboxDescriptorSets();
//        TransparentLayers[i].createSpotLightingDescriptorPool();
//        TransparentLayers[i].createSpotLightingDescriptorSets();
//    }
}

void VkApplication::updateDescriptorSets()
{
    DeferredGraphics.updateBaseDescriptorSets(nullptr);
    DeferredGraphics.updateSkyboxDescriptorSets();
    DeferredGraphics.updateSpotLightingDescriptorSets();

//    for(uint32_t i=0;i<TransparentLayersCount;i++){
//        TransparentLayers[i].updateBaseDescriptorSets(nullptr);
//        TransparentLayers[i].updateSkyboxDescriptorSets();
//        TransparentLayers[i].updateSpotLightingDescriptorSets();
//    }
}

void VkApplication::createCommandBuffers()
{
    createCommandBuffer();
    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        updateCommandBuffer(imageIndex);

    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++)
        DeferredGraphics.updateSpotLightCmd(imageIndex);

//    for(size_t imageIndex=0;imageIndex<imageCount;imageIndex++){
//        for(uint32_t i=0;i<TransparentLayersCount;i++){
//            TransparentLayers[i].updateSpotLightCmd(imageIndex);
//        }
//    }
}
    void VkApplication::createCommandBuffer()
    {
        commandBuffers.resize(imageCount);

        VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = commandPool;                                 //дескриптор ранее созданного командного пула
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;                   //задаёт уровень командных буферов, которые вы хотите выделить
            allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();     //задаёт число командных буферов

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate command buffers!");
    }

void VkApplication::updateCommandBuffer(uint32_t imageIndex)
{
    vkResetCommandBuffer(commandBuffers[imageIndex],0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                                            //поле для передачи информации о том, как будет использоваться этот командный буфер (смотри страницу 102)
        beginInfo.pInheritanceInfo = nullptr;                           //используется при начале вторичного буфера, для того чтобы определить, какие состояния наследуются от первичного командного буфера, который его вызовет
    if (vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording command buffer!");

    DeferredGraphics.render(imageIndex,commandBuffers[imageIndex]);

    SSLR.render(imageIndex,commandBuffers[imageIndex]);
    SSAO.render(imageIndex,commandBuffers[imageIndex]);

        VkImageSubresourceRange ImageSubresourceRange{};
            ImageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ImageSubresourceRange.baseMipLevel = 0;
            ImageSubresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            ImageSubresourceRange.baseArrayLayer = 0;
            ImageSubresourceRange.layerCount = 1;
        VkClearColorValue clearColorValue{};
            clearColorValue.uint32[0] = 0;
            clearColorValue.uint32[1] = 0;
            clearColorValue.uint32[2] = 0;
            clearColorValue.uint32[3] = 0;

        std::vector<VkImage> blitImages(PostProcessing.getBlitAttachments().size());
        blitImages[0] = DeferredGraphics.getDeferredAttachments().bloom->image[imageIndex];
        for(size_t i=1;i<PostProcessing.getBlitAttachments().size();i++){
            blitImages[i] = PostProcessing.getBlitAttachments()[i-1].image[imageIndex];
        }
        VkImage blitBufferImage = PostProcessing.getBlitAttachment().image[imageIndex];
        uint32_t width = PostProcessing.SwapChainImageExtent().width;
        uint32_t height = PostProcessing.SwapChainImageExtent().height;
        float blitFactor = PostProcessing.getBlitFactor();

        for(uint32_t k=0;k<PostProcessing.getBlitAttachments().size();k++){
            transitionImageLayout(&commandBuffers[imageIndex],blitBufferImage,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_REMAINING_MIP_LEVELS);
            vkCmdClearColorImage(commandBuffers[imageIndex],blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ,&clearColorValue,1,&ImageSubresourceRange);
            blitDown(&commandBuffers[imageIndex],blitImages[k],blitBufferImage,width,height,blitFactor);
            transitionImageLayout(&commandBuffers[imageIndex],blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);
            Filter.render(imageIndex,commandBuffers[imageIndex],k);
        }
        for(uint32_t k=0;k<PostProcessing.getBlitAttachments().size();k++)
            transitionImageLayout(&commandBuffers[imageIndex],PostProcessing.getBlitAttachments()[k].image[imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);

    PostProcessing.render(imageIndex,commandBuffers[imageIndex]);

    if (vkEndCommandBuffer(commandBuffers[imageIndex]) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
}

void VkApplication::createSyncObjects()
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
    imagesInFlight.resize(imageCount, VK_NULL_HANDLE);

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

VkResult VkApplication::checkNextFrame()
{
    VkResult result = vkAcquireNextImageKHR(device, PostProcessing.SwapChain(), UINT64_MAX , imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);    //Получить индекс следующего доступного презентабельного изображения

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        return result;

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)                                       //если нет следующего изображения
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);       //ждём
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    return result;
}

VkResult VkApplication::drawFrame()
{
    updateCmd(imageIndex);
    updateUbo(imageIndex);

    VkSemaphore                         waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags                waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore                         signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    std::vector<VkCommandBuffer>        commandbufferSet;

    DeferredGraphics.getSpotLightCommandbuffers(&commandbufferSet,imageIndex);
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
    VkSwapchainKHR swapChains[] = {PostProcessing.SwapChain()};
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
    void VkApplication::updateCmd(uint32_t imageIndex)
    {
        if(worldCmd.enable)
        {
            updateCommandBuffer(imageIndex);
            if((++worldCmd.frames)==imageCount)
                worldCmd.enable = false;
        }
        if(lightsCmd.enable)
        {
            DeferredGraphics.updateSpotLightCmd(imageIndex);
            if((++lightsCmd.frames)==imageCount)
                lightsCmd.enable = false;
        }
    }
    void VkApplication::updateUbo(uint32_t imageIndex)
    {
        if(worldUbo.enable)
        {
            updateUniformBuffer(imageIndex);
            if((++worldUbo.frames)==imageCount)
                worldUbo.enable = false;
        }
        if(lightsUbo.enable)
        {
            DeferredGraphics.updateSpotLightUbo(imageIndex);
            if((++lightsUbo.frames)==imageCount)
                lightsUbo.enable = false;
        }
    }
        void VkApplication::updateUniformBuffer(uint32_t imageIndex)
        {
            DeferredGraphics.updateUniformBuffer(imageIndex);
            DeferredGraphics.updateSkyboxUniformBuffer(imageIndex);
            DeferredGraphics.updateObjectUniformBuffer(imageIndex);
        }

void VkApplication::destroyGraphics()
{
    DeferredGraphics.destroy();
    Filter.destroy();
    SSAO.destroy();
    SSLR.destroy();
    PostProcessing.destroy();
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].destroy();
}

void VkApplication::freeCommandBuffers()
{
    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()),commandBuffers.data());
    commandBuffers.clear();
}

void VkApplication::VkApplication::cleanup()
{
    DeferredGraphics.destroyEmptyTexture();
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].destroyEmptyTexture();

    destroyGraphics();
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
    void VkApplication::DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
    {
        //VkDebugUtilsMessengerEXT Объект также должен быть очищен с помощью вызова vkDestroyDebugUtilsMessengerEXT
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if(func != nullptr)    func(instance, debugMessenger, pAllocator);
    }

uint32_t                            VkApplication::getImageCount(){return imageCount;}
uint32_t                            VkApplication::getImageIndex(){return imageIndex;}
uint32_t                            VkApplication::getCurrentFrame(){return currentFrame;}

void                                VkApplication::resetCmdLight(){lightsCmd.enable = true; lightsCmd.frames = 0;}
void                                VkApplication::resetCmdWorld(){worldCmd.enable = true; worldCmd.frames = 0;}
void                                VkApplication::resetUboLight(){lightsUbo.enable = true; lightsUbo.frames = 0;}
void                                VkApplication::resetUboWorld(){worldUbo.enable = true; worldUbo.frames = 0;}
void                                VkApplication::setExternalPath(const std::string &path){ExternalPath = path;}

void                                VkApplication::setEmptyTexture(std::string ZERO_TEXTURE){
    DeferredGraphics.setEmptyTexture(ZERO_TEXTURE);

//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].setEmptyTexture(ZERO_TEXTURE);
}

void                                VkApplication::setCameraObject(camera* cameraObject){
    DeferredGraphics.setCameraObject(cameraObject);

//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].setCameraObject(cameraObject);
}

void                                VkApplication::createModel(gltfModel *pModel){
    pModel->loadFromFile(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool,1.0f);
    pModel->createDescriptorPool(&device);
    pModel->createDescriptorSet(&device,DeferredGraphics.getEmptyTexture());
}

void                                VkApplication::destroyModel(gltfModel* pModel){
    pModel->destroy(&device);
}

void                                VkApplication::addLightSource(spotLight* lightSource)
{
    if(lightSource->getTexture()){
        lightSource->getTexture()->createTextureImage(&physicalDevices.at(physicalDeviceNumber).device,&device,&graphicsQueue,&commandPool);
        lightSource->getTexture()->createTextureImageView(&device);
        lightSource->getTexture()->createTextureSampler(&device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    }
    lightSource->createUniformBuffers(&physicalDevices.at(physicalDeviceNumber).device,&device,imageCount);
    lightSource->createShadow(&physicalDevices.at(physicalDeviceNumber).device,&device,&physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber),imageCount,ExternalPath);
    lightSource->updateShadowDescriptorSets();
    if(lightSource->isShadowEnable()){
        lightSource->createShadowCommandBuffers();
    }

    lightSource->createDescriptorPool(&device, imageCount);
    lightSource->createDescriptorSets(&device, imageCount);
    lightSource->updateDescriptorSets(&device, imageCount,DeferredGraphics.getEmptyTexture());

    DeferredGraphics.addSpotLightSource(lightSource);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].addSpotLightSource(lightSource);
}
void                                VkApplication::removeLightSource(spotLight* lightSource)
{
    if(lightSource->getTexture()){
        lightSource->getTexture()->destroy(&device);
    }
    lightSource->destroyUniformBuffers(&device);
    lightSource->destroy(&device);

    DeferredGraphics.removeSpotLightSource(lightSource);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].removeSpotLightSource(lightSource);
}

void                                VkApplication::bindBaseObject(object* newObject)
{
    newObject->createUniformBuffers(&physicalDevices.at(physicalDeviceNumber).device,&device,imageCount);
    newObject->createDescriptorPool(&device,imageCount);
    newObject->createDescriptorSet(&device,imageCount);
    DeferredGraphics.bindBaseObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindBaseObject(newObject);
}
void                                VkApplication::bindBloomObject(object* newObject)
{
    newObject->createUniformBuffers(&physicalDevices.at(physicalDeviceNumber).device,&device,imageCount);
    newObject->createDescriptorPool(&device,imageCount);
    newObject->createDescriptorSet(&device,imageCount);
    DeferredGraphics.bindBloomObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindBloomObject(newObject);
}
void                                VkApplication::bindOneColorObject(object* newObject)
{
    newObject->createUniformBuffers(&physicalDevices.at(physicalDeviceNumber).device,&device,imageCount);
    newObject->createDescriptorPool(&device,imageCount);
    newObject->createDescriptorSet(&device,imageCount);
    DeferredGraphics.bindOneColorObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindOneColorObject(newObject);
}
void                                VkApplication::bindStencilObject(object* newObject, float lineWidth, glm::vec4 lineColor)
{
    newObject->setStencilEnable(false);
    newObject->setStencilWidth(lineWidth);
    newObject->setStencilColor(lineColor);
    newObject->createUniformBuffers(&physicalDevices.at(physicalDeviceNumber).device,&device,imageCount);
    newObject->createDescriptorPool(&device,imageCount);
    newObject->createDescriptorSet(&device,imageCount);
    DeferredGraphics.bindStencilObject(newObject);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindStencilObject(newObject);
}
void                                VkApplication::bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH)
{
    newObject->createUniformBuffers(&physicalDevices.at(physicalDeviceNumber).device,&device,imageCount);
    DeferredGraphics.bindSkyBoxObject(newObject,TEXTURE_PATH);
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].bindSkyBoxObject(newObject,TEXTURE_PATH);
}

bool                                VkApplication::removeBaseObject(object* object)
{
    object->destroy(&device);
    object->destroyUniformBuffers(&device);
    return DeferredGraphics.removeBaseObject(object);
}
bool                                VkApplication::removeBloomObject(object* object)
{
    object->destroy(&device);
    object->destroyUniformBuffers(&device);
    return DeferredGraphics.removeBloomObject(object);
}
bool                                VkApplication::removeOneColorObject(object* object)
{
    object->destroy(&device);
    object->destroyUniformBuffers(&device);
    return DeferredGraphics.removeOneColorObject(object);
}
bool                                VkApplication::removeStencilObject(object* object)
{
    object->destroy(&device);
    object->destroyUniformBuffers(&device);
    return DeferredGraphics.removeStencilObject(object);
}
bool                                VkApplication::removeSkyBoxObject(object* object)
{
    object->destroyUniformBuffers(&device);
    return DeferredGraphics.removeSkyBoxObject(object);
}

void                                VkApplication::removeBinds(){
    DeferredGraphics.removeBinds();
//    for(uint32_t i=0;i<TransparentLayersCount;i++)
//        TransparentLayers[i].removeBinds();
}

void                                VkApplication::setMinAmbientFactor(const float& minAmbientFactor){
    DeferredGraphics.setMinAmbientFactor(minAmbientFactor);
    for(uint32_t i=0;i<TransparentLayersCount;i++)
        TransparentLayers[i].setMinAmbientFactor(minAmbientFactor);
}

void                                VkApplication::updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition){
    DeferredGraphics.updateStorageBuffer(currentImage,mousePosition);
}
uint32_t                            VkApplication::readStorageBuffer(uint32_t currentImage){
    return DeferredGraphics.readStorageBuffer(currentImage);
}

void                                VkApplication::deviceWaitIdle()
{
    vkDeviceWaitIdle(device);
}
