#include "vulkanCore.h"
#include "operations.h"
#include "transformational/object.h"
#include "transformational/group.h"
#include "transformational/camera.h"
#include "transformational/light.h"
#include "transformational/gltfmodel.h"

std::vector<std::string> SKYBOX = {
    "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\skybox\\left.jpg",
    "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\skybox\\right.jpg",
    "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\skybox\\front.jpg",
    "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\skybox\\back.jpg",
    "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\skybox\\top.jpg",
    "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\skybox\\bottom.jpg"
};
std::string ZERO_TEXTURE = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\0.png";
std::string ZERO_TEXTURE_WHITE = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\texture\\1.png";

void VkApplication::VkApplication::initWindow()
{
    glfwInit();                                                             //инициализация библиотеки GLFW

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);                           //указывает не создавать контекст OpenGL (GLFW изначально был разработан для создания контекста OpenGL,)

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);   //инициализация собственного окна
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

}
    void VkApplication::VkApplication::framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        static_cast<void>(width);
        static_cast<void>(height);
        auto app = reinterpret_cast<VkApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

void VkApplication::VkApplication::initVulkan()
{
    createInstance();               //создание экземпляра, инициализация библиотеки vulkan происходит при создании экземпляра
    setupDebugMessenger();          //загрузить отладочный мессенджер
    createSurface();                //создать поверхность окна для приложения Vullkan

    pickPhysicalDevice();           //выбрать физические устройства
    createLogicalDevice();          //создать логические устройства

    createCommandPool();    

    createTextures();
    loadModel();
    createObjects();

    createLight();

    createGraphics();

    updateLight();
    updateObjectsUniformBuffers();

    createDescriptors();

    createCommandBuffers();

    createSyncObjects();
}

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

    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

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
    {                                                                                   //параметр pAllocator указывает на аллокатор памяти CPU который ваше приложение может передать для управления используемой Vulkan памятью
        throw std::runtime_error("failed to create instance!");                         //при передачи nullptr Vulkan будет использовать свой собственный внутренний аллокатор
    }
}
    //создадим getRequiredExtensions функцию,
    //которая будет возвращать требуемый список расширений в зависимости от того,
    //включены ли уровни проверки или нет
    std::vector<const char*> VkApplication::getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if(enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); //макрос, который равен буквальной строке «VK_EXT_debug_utils»
        }

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
            {
                return false;
            }
        }

        return true;
    }

//===================================DebugMessenger====================================//

void VkApplication::VkApplication::setupDebugMessenger()
{
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
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
        if (func != nullptr) {
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        } else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

//===========================Surface==========================//

void VkApplication::createSurface()
{
    /*Объект, в который осуществляется рендеринг для показа пользователю, называется поверхностью(surface) и представлен при помощи дескриптора VkSurfaceKHR.
     * Это специальный объект, вводимый расширением VK_KHR_surface. Это расширение вводит общую функциональнать для работы с поверхностями, которая в дальнейшем
     * адаптируется под каждую платформу лоя получения платформенно-зависимого интерфейса для связи с поверхостью окна*/

    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)         //Эта функция создает поверхность Vulkan для указанного окна window.
    {
            throw std::runtime_error("failed to create window surface!");
    }
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
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices)
    {
        std::vector<QueueFamilyIndices> indices = findQueueFamilies(device, surface);
        if (indices.size()!=0 && isDeviceSuitable(device,surface,deviceExtensions))
        {
            struct physicalDevice currentDevice = {device,indices};
            physicalDevices.push_back(currentDevice);
        }
    }

    if(physicalDevices.size()!=0)
    {
        physicalDevice = physicalDevices.at(0).device;
        indices = physicalDevices.at(0).indices.at(0);
        msaaSamples = getMaxUsableSampleCount(physicalDevice);
    }

    if (physicalDevice == VK_NULL_HANDLE)                               //если не нашли ни одного поддерживаемого устройства, то выводим соответствующую ошибку
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void VkApplication::createLogicalDevice()
{
    /* Логическое устройство - это программная абстракция физического устройсва, сконфигурированная в соответствии
     * с тем, как задано приложение. После выбора физического устройства вашему приложению необходимо создать
     * соответствующее ему логическое устройство.*/

    QueueFamilyIndices indices = this->indices;     //при помощи этой функции ищем первое подходящее семейство очередей

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;              //массив очередей
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};  //массив очередей

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
    deviceFeatures.samplerAnisotropy = VK_TRUE;             //анизотропная фильтрация
    deviceFeatures.independentBlend = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;
    deviceFeatures.imageCubeArray = VK_TRUE;

    //создаём информацию о новом логическом устройстве
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());   //количество очередей в массиве
    createInfo.pQueueCreateInfos = queueCreateInfos.data();                             //массив очередей соответственно
    createInfo.pEnabledFeatures = &deviceFeatures;                                      //поддерживаемые дополнительные возможности устройства
    //Для использования swapchain необходимо сначала включить VK_KHR_swapchain расширение.
    //Для включения расширения требуется лишь небольшое изменение структуры создания логического устройства:
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());  //задаём количество расширений (оно у нас одно)
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();                       //передём имена этих расширений

    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else
    {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) //создание логического устройства
    {
        throw std::runtime_error("failed to create logical device!");
    }

    //Получение дескрипторов очереди из найденного семейства очередей от выбранного устройства
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

void VkApplication::createCommandPool()
{
    /* Главной целью очереди является выполнение работы от имени вашего приложения.
     * Работа/задания представлены как последовательность команд, которые записываются
     * в командные буферы. Ваше приложение создаёт командные буферы, одержащие задания,
     * которые необходимо выполнить, и передает (submit) их в одну из очередей для выполения.
     * Прежде чем вы можете запоминать какие-либо  команды, вам нужно создать командный буфер.
     * Командные буферы не создаются явно, а выделяются из пулов.*/

    QueueFamilyIndices queueFamilyIndices = this->indices;  //находим индексы очередей

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();      //задаёт семейство очередей, в которые будет передаваться созданные командные буферы
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;           //задаёт флаги, определяющие поведение пула и командных буферов, выделяемых из него

    commandPool.resize(COMMAND_POOLS);
    commandBuffers.resize(COMMAND_POOLS);

    for(size_t i=0;i<COMMAND_POOLS;i++)
    {
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool.at(i)) != VK_SUCCESS) //создание пула команд
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }
}

void VkApplication::createTextures()
{
    skybox = new cubeTexture(this,SKYBOX);
    skybox->setMipLevel(0.0f);
    skybox->createTextureImage();
    skybox->createTextureImageView();
    skybox->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});

    emptyTexture = new texture(this,ZERO_TEXTURE);
    emptyTexture->createTextureImage();
    emptyTexture->createTextureImageView();
    emptyTexture->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});

    emptyTextureW = new texture(this,ZERO_TEXTURE_WHITE);
    emptyTextureW->createTextureImage();
    emptyTextureW->createTextureImageView();
    emptyTextureW->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
}

void VkApplication::loadModel()
{
    gltfModel.push_back(new struct gltfModel);
    gltfModel.at(gltfModel.size()-1)->loadFromFile("C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\model\\glb\\Bee.glb",this,1.0f);
    gltfModel.push_back(new struct gltfModel);
    gltfModel.at(gltfModel.size()-1)->loadFromFile("C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\model\\glb\\Bee.glb",this,1.0f);
    gltfModel.push_back(new struct gltfModel);
    gltfModel.at(gltfModel.size()-1)->loadFromFile("C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\model\\glb\\Box.glb",this,1.0f);
    gltfModel.push_back(new struct gltfModel);
    gltfModel.at(gltfModel.size()-1)->loadFromFile("C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\model\\glTF\\Sponza\\Sponza.gltf",this,1.0f);
    gltfModel.push_back(new struct gltfModel);
    gltfModel.at(gltfModel.size()-1)->loadFromFile("C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\model\\glb\\Duck.glb",this,1.0f);
    gltfModel.push_back(new struct gltfModel);
    gltfModel.at(gltfModel.size()-1)->loadFromFile("C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\model\\glb\\RetroUFO.glb",this,1.0f);
}

void VkApplication::createObjects()
{
    uint32_t index=0;
    object3D.push_back( new object(this,{gltfModel.at(0),&Graphics.PipeLine(),&Graphics.PipelineLayout(),emptyTexture}) );
    object3D.at(index)->translate(glm::vec3(3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    index++;

    object3D.push_back( new object(this,{gltfModel.at(1),&Graphics.PipeLine(),&Graphics.PipelineLayout(),emptyTexture}) );
    object3D.at(index)->translate(glm::vec3(-3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    object3D.at(index)->animationTimer = 1.0f;
    object3D.at(index)->animationIndex = 1;
    index++;

    object3D.push_back( new object(this,{gltfModel.at(3),&Graphics.PipeLine(),&Graphics.PipelineLayout(),emptyTexture}) );
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(3.0f,3.0f,3.0f));
    index++;

    object3D.push_back( new object(this,{gltfModel.at(4),&Graphics.PipeLine(),&Graphics.PipelineLayout(),emptyTexture}) );
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(1.0f,1.0f,1.0f));
    object *Duck = object3D.at(index);
    index++;

    object3D.push_back( new object(this,{gltfModel.at(5),&Graphics.PipeLine(),&Graphics.PipelineLayout(),emptyTexture}) );
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(1.0f,1.0f,1.0f));
    object *UFO1 = object3D.at(index);
    index++;

    object3D.push_back( new object(this,{gltfModel.at(5),&Graphics.PipeLine(),&Graphics.PipelineLayout(),emptyTexture}) );
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(1.0f,1.0f,1.0f));
    object *UFO2 = object3D.at(index);
    index++;

    object3D.push_back( new object(this,{gltfModel.at(2),&Graphics.BloomSpriteGraphicsPipeline(),&Graphics.BloomSpritePipelineLayout(),emptyTextureW}) );
    object *Box = object3D.at(index);
    index++;

    skyboxObject = new object(this,{gltfModel.at(2),nullptr,nullptr,nullptr});
    skyboxObject->scale(glm::vec3(200.0f,200.0f,200.0f));

    groups.push_back(new group);
    groups.at(0)->translate(glm::vec3(0.0f,0.0f,5.0f));
    groups.at(0)->addObject(Box);

    groups.push_back(new group);
    groups.at(1)->addObject(Duck);

    groups.push_back(new group);
    groups.at(2)->translate(glm::vec3(5.0f,0.0f,5.0f));
    groups.at(2)->addObject(UFO1);

    groups.push_back(new group);
    groups.at(3)->translate(glm::vec3(-5.0f,0.0f,5.0f));
    groups.at(3)->addObject(UFO2);

    cam = new camera;
    cam->translate(glm::vec3(0.0f,0.0f,10.0f));
}

void VkApplication::createLight()
{
    int index = 0;
    lightPoint.push_back(new light<pointLight>(this,lightSource));
    lightPoint.at(0)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(lightPoint.at(0));
    index +=6;

    glm::mat4x4 Proj;

    lightSource.push_back(new light<spotLight>(this));
    lightSource.at(index)->setCommandPoolsCount(1);
    lightSource.at(index)->createCommandPool();
    lightSource.at(index)->createShadowSampler();
    Proj = glm::perspective(glm::radians(90.0f), (float) lightSource.at(index)->getWidth()/lightSource.at(index)->getHeight(), 0.1f, 40.0f);
    Proj[1][1] *= -1;
    lightSource.at(index)->createLightPVMatrix(Proj,glm::mat4x4(1.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,0.0f,1.0f,0.0f));
    groups.at(2)->addObject(lightSource.at(index));
    index++;

    lightSource.push_back(new light<spotLight>(this));
    lightSource.at(index)->setCommandPoolsCount(1);
    lightSource.at(index)->createCommandPool();
    lightSource.at(index)->createShadowSampler();
    Proj = glm::perspective(glm::radians(90.0f), (float) lightSource.at(index)->getWidth()/lightSource.at(index)->getHeight(), 0.1f, 40.0f);
    Proj[1][1] *= -1;
    lightSource.at(index)->createLightPVMatrix(Proj,glm::mat4x4(1.0f));
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,0.0f,0.0f,0.0f));
    groups.at(3)->addObject(lightSource.at(index));
    index++;
}

void VkApplication::createGraphics()
{    
    PostProcessing.setApplication(this);
    PostProcessing.setMSAASamples(msaaSamples);

    PostProcessing.createSwapChain();
    PostProcessing.createImageViews();
    PostProcessing.createRenderPass();
    PostProcessing.createDescriptorSetLayout();
    PostProcessing.createGraphicsPipeline();
    PostProcessing.createFramebuffers();

    imageCount = PostProcessing.ImageCount();

    Graphics.setApplication(this);
    Graphics.setMSAASamples(msaaSamples);
    Graphics.setImageProp(imageCount,PostProcessing.SwapChainImageFormat(),PostProcessing.SwapChainExtent());
    Graphics.setEmptyTexture(emptyTexture);
    Graphics.setSkyboxTexture(skybox);

    Graphics.createColorAttachments();
    Graphics.createDepthAttachment();
    Graphics.createAttachments();
    Graphics.createUniformBuffers();
    Graphics.createSkyboxUniformBuffers();
    Graphics.createDrawRenderPass();
    Graphics.createDescriptorSetLayout();
    Graphics.createSkyboxDescriptorSetLayout();
    Graphics.createGraphicsPipeline();
    Graphics.createSkyBoxPipeline();
    Graphics.createGodRaysGraphicsPipeline();
    Graphics.createBloomSpriteGraphicsPipeline();
    Graphics.createFramebuffers();
}

void VkApplication::updateLight()
{
    for(size_t i=0; i<lightSource.size();i++)
    {
        lightSource.at(i)->setImageCount(imageCount);
        lightSource.at(i)->createShadowRenderPass();
        lightSource.at(i)->createShadowImage();
        lightSource.at(i)->createShadowImageView();
        lightSource.at(i)->createShadowMapFramebuffer();
        lightSource.at(i)->createUniformBuffers();
        lightSource.at(i)->createShadowDescriptorSetLayout();
        lightSource.at(i)->createShadowPipeline();
        lightSource.at(i)->createShadowDescriptorPool();
        lightSource.at(i)->createShadowDescriptorSets();
    }
}

void VkApplication::updateObjectsUniformBuffers()
{
    for(size_t i=0;i<object3D.size();i++)
    {
        object3D.at(i)->createUniformBuffers(imageCount);
    }
}

void VkApplication::createDescriptors()
{
    PostProcessing.createDescriptorPool(Graphics.getAttachments());
    PostProcessing.createDescriptorSets(Graphics.getAttachments());

    Graphics.createDescriptorPool(object3D);
    Graphics.createDescriptorSets(lightSource,object3D);
    Graphics.createSkyboxDescriptorPool();
    Graphics.createSkyboxDescriptorSets();
}

void VkApplication::createCommandBuffers()
{
    updateCmdLight = true;
    updateCmdWorld = true;

    for(size_t i=0;i<lightSource.size();i++)
    {
        for(size_t j=0;j<commandPool.size();j++)
        {
            lightSource.at(i)->createShadowCommandBuffers(j);
        }
    }

    for(size_t i=0;i<commandPool.size();i++)
    {
        createCommandBuffer(i);
    }
}

void VkApplication::createCommandBuffer(uint32_t number)
{
    /*После того как мы получили пул для создания командных буферов, мы можем
     * получить новый буфер при помощи функции vkAllocateCommandBuffers()*/

    commandBuffers[number].resize(imageCount);

    //сперва заполним структуру содержащую информацию для создания командного буфера
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool[number];                                 //дескриптор ранее созданного командного пула
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;                              //задаёт уровень командных буферов, которые вы хотите выделить
    allocInfo.commandBufferCount = (uint32_t) commandBuffers[number].size();     //задаёт число командных буферов

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers[number].data()) != VK_SUCCESS) //получаем командные буферы
    {   //если вызов успешен, то он вернёт VK_SUCCESS и поместит дескрипторы созданных буферов в массив, адресс которого содержится в pCommandBuffers
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void VkApplication::updateCommandBuffer(uint32_t number, uint32_t i)
{
        /* Прежде чем вы сможете начать записывать команды в командный буфер, вам
         * нужно начать командный буфер, т.е. просто сбростить к начальному состоянию.
         * Для этого выозвите функцию vkBeginCommandBuffer*/

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                                            //поле для передачи информации о том, как будет использоваться этот командный буфер (смотри страницу 102)
        beginInfo.pInheritanceInfo = nullptr;                           //используется при начале вторичного буфера, для того чтобы определить, какие состояния наследуются от первичного командного буфера, который его вызовет
        if (vkBeginCommandBuffer(commandBuffers[number][i], &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

            Graphics.render(commandBuffers[number],i,object3D,*skyboxObject);
            PostProcessing.render(commandBuffers[number],i);

        if (vkEndCommandBuffer(commandBuffers[number][i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
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

    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(imageCount, VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void VkApplication::VkApplication::mainLoop()
{
    static auto pastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        frameTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - pastTime).count();

        if(fpsLock){if(fps<1.0f/frameTime){continue;}}
        pastTime = currentTime;

        updateAnimations(animate);
        glfwPollEvents();
        mouseEvent();
        drawFrame();
    }

    vkDeviceWaitIdle(device);
}
    void VkApplication::updateAnimations(bool animate)
    {
        if(animate)
        {
            for(size_t j=0;j<object3D.size();j++)
            {
                if(object3D[j]->getModel()->animations.size() > 0)
                {
                    object3D[j]->animationTimer += frameTime;
                    if (object3D[j]->animationTimer > object3D[j]->getModel()->animations[object3D[j]->animationIndex].end)
                    {
                        object3D[j]->animationTimer -= object3D[j]->getModel()->animations[object3D[j]->animationIndex].end;
                    }
                    object3D[j]->getModel()->updateAnimation(object3D[j]->animationIndex, object3D[j]->animationTimer);
                }
            }
        }
    }
    void VkApplication::drawFrame()
    {
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, PostProcessing.SwapChain(), UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);    //Получить индекс следующего доступного презентабельного изображения

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)                                       //если нет слежующего изображения
        {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);       //ждём
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        updateCmd(imageIndex);

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};

        VkCommandBuffer commandbufferSet[lightSource.size()+1];
        for(size_t i=0;i<lightSource.size();i++)
        {
            commandbufferSet[i] = lightSource[i]->getCommandBuffer(currentBuffer)[imageIndex];
        }
        commandbufferSet[lightSource.size()] = commandBuffers[currentBuffer][imageIndex];

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = lightSource.size()+1;
        submitInfo.pCommandBuffers = commandbufferSet;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);                                        //восстановить симофоры в несингнальное положение

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)   //отправляет последовательность семафоров или командных буферов в очередь
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        VkSwapchainKHR swapChains[] = {PostProcessing.SwapChain()};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);                                                  //Поставить изображение в очередь для презентации

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        if(imageIndex==PostProcessing.Framebuffers().size()-1)
        {
            currentBuffer = (currentBuffer + 1) % COMMAND_POOLS;
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        //vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    }
    void VkApplication::updateCmd(uint32_t imageIndex)
    {
        if(updateCmdWorld)
        {
            updateUniformBuffer(imageIndex);
            updateCommandBuffer((currentBuffer + 1) % COMMAND_POOLS,imageIndex);
            updatedWorldFrames++;
            if(updatedWorldFrames==COMMAND_POOLS*imageCount)
            {
                updateCmdWorld = false;
            }
        }
        if(updateCmdLight)
        {
            for(size_t i=0;i<lightSource.size();i++)
            {
                lightSource[i]->updateShadowUniformBuffer(imageIndex);
                lightSource[i]->updateShadowCommandBuffers((currentBuffer + 1) % COMMAND_POOLS,imageIndex,object3D);
            }
            updatedLightFrames++;
            if(updatedLightFrames==COMMAND_POOLS*imageCount)
            {
                updateCmdLight = false;
            }
        }
    }
        void VkApplication::updateUniformBuffer(uint32_t currentImage)
        {
            Graphics.updateUniformBuffer(currentImage, cam, skyboxObject);

            for(size_t i=0;i<object3D.size();i++)
            {
                object3D.at(i)->updateUniformBuffer(currentImage);
            }
        }
        void VkApplication::recreateSwapChain()
        {
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            while (width == 0 || height == 0)
            {
                glfwGetFramebufferSize(window, &width, &height);
                glfwWaitEvents();
            }

            vkDeviceWaitIdle(device);

            cleanupSwapChain();

            createGraphics();

            updateLight();
            updateObjectsUniformBuffers();

            createDescriptors();

            createCommandBuffers();
        }
            void VkApplication::cleanupSwapChain()
            {
                for(size_t i=0;i<lightSource.size();i++)
                {
                    lightSource.at(i)->cleanup();
                }

                Graphics.destroy();
                for(size_t i=0;i<object3D.size();i++)
                {
                    object3D.at(i)->destroyUniformBuffers();
                    object3D.at(i)->destroyDescriptorPools();
                }
                PostProcessing.destroy();

                for(size_t i = 0; i< commandPool.size();i++)
                {
                    vkFreeCommandBuffers(device, commandPool.at(i), static_cast<uint32_t>(commandBuffers.at(i).size()),commandBuffers.at(i).data());
                }
            }


void VkApplication::VkApplication::cleanup()
{
    cleanupSwapChain();
    for(size_t i=0;i<lightSource.size();i++)
    {
        delete lightSource.at(i);
    }

    for(size_t i=0;i<lightPoint.size();i++)
    {
        delete lightPoint.at(i);
    }

    for(size_t i =0 ;i<textures.size();i++)
    {
        textures.at(i)->destroy();
        delete textures.at(i);
    }

    skybox->destroy();
    delete skybox;
    emptyTexture->destroy();
    delete emptyTexture;
    emptyTextureW->destroy();
    delete emptyTextureW;

    for (size_t i =0 ;i<object3D.size();i++)
    {
        delete object3D.at(i);
    }

    for (size_t i =0 ;i<gltfModel.size();i++)
    {
        gltfModel.at(i)->destroy(gltfModel.at(i)->app->getDevice());
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    for(size_t i = 0; i< commandPool.size();i++)
    {
        vkDestroyCommandPool(device, commandPool.at(i), nullptr);
    }

    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers)
    {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();
}
    void VkApplication::DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
    {
        //VkDebugUtilsMessengerEXT Объект также должен быть очищен с помощью вызова vkDestroyDebugUtilsMessengerEXT
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr)
        {
            func(instance, debugMessenger, pAllocator);
        }
    }

VkPhysicalDevice                    & VkApplication::getPhysicalDevice(){return physicalDevice;}
VkDevice                            & VkApplication::getDevice(){return device;}
VkQueue                             & VkApplication::getGraphicsQueue(){return graphicsQueue;}
std::vector<VkCommandPool>          & VkApplication::getCommandPool(){return commandPool;}
VkSurfaceKHR                        & VkApplication::getSurface(){return surface;}
GLFWwindow                          & VkApplication::getWindow(){return *window;}
QueueFamilyIndices                  & VkApplication::getQueueFamilyIndices(){return indices;}
