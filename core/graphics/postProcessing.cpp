#include "graphics.h"
#include "core/operations.h"

postProcessing::postProcessing()
{

}

void postProcessing::setApplication(VkApplication * app)
{
    this->app = app;
}

void postProcessing::setMSAASamples(VkSampleCountFlagBits msaaSamples)
{
    this->msaaSamples = msaaSamples;
}

void postProcessing::destroy()
{
    vkDestroyPipeline(app->getDevice(), graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), pipelineLayout,nullptr);

    vkDestroyRenderPass(app->getDevice(), renderPass, nullptr);

    for(size_t i=0; i<swapChainAttachments.size(); i++)
    {
        for(size_t image=0; image <imageCount;image++)
        {
            vkDestroyImageView(app->getDevice(),swapChainAttachments.at(i).imageView[image],nullptr);
        }
    }

    vkDestroySwapchainKHR(app->getDevice(), swapChain, nullptr);

    for(size_t i = 0; i< framebuffers.size();i++)
    {
        vkDestroyFramebuffer(app->getDevice(), framebuffers[i],nullptr);
    }

    vkDestroyDescriptorSetLayout(app->getDevice(), descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), descriptorPool, nullptr);
}

//Создание цепочки обмена
void postProcessing::createSwapChain()
{
    /* Независимо от платформы, на которой вы выполняете свой код, получающийся
     * дескриптор VkSurfaceKHR соответствует тому, как Vulkan видит окно. Для того
     * чтобы в реальности показать что-то на этой поверхности, необходимо создать
     * специальное изображение, которое будет использоваться для хранения данных
     * в окне.  На большинстве платформ этот тип изображения либо принадлежит, либо
     * тесно связан с оконной системой, поэтому вместо создания нормального изображения
     * Vulkan мы будем использовать второй обьект, называемый список показа (swap chain),
     * для управенления одним или несколькими изображениями.
     * Списки показа спользуются для того, чтобы попростить оконную систему создать
     * одно или несколько изображений, коорые будут использоваться для показа в поверхность Vulkan.
     * Эта функциональность предоставляется при помощи расширения VK_KHR_swapchain.
     * Каждыф список показа управляется набором изображений, обычно, организованны в виде кольцевого буфера.
     * Приложение может попросить сисок показа дать следующее доступное изображение, осуществить
     * рендеринг в него и отдать его обратно списку показа как готовое к показу. За счёт
     * организации изображений в виде кольцевого буфера или очереди одно изображение может
     * показываться на экране, в то время как в другое осуществляется рендеринг*/

    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(app->getPhysicalDevice(),app->getSurface());           //здест происходит запрос поддерживаемы режимов и форматов которые в следующий строчках передаются в соответствующие переменные через фукцнии

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);                                   //задаём поддерживаемый формат
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);                                    //смотри ниже
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);                                                    //задаём размер изображения в списке показа в пикселях

    imageCount = swapChainSupport.capabilities.minImageCount + 1;                                                           //запрос на поддержк уминимального количества числа изображений, число изображений равное 2 означает что один буфер передний, а второй задний
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)        //в первом условии мы проверяем доступно ли нам вообще какое-то количество изображений
    {                                                                                                                       //и проверяем не совпадает ли максимальное число изображений с минимальным
        imageCount = swapChainSupport.capabilities.maxImageCount;                                                           //присываиваем максимальное значение
    }

    //Создаём соответствующую структуру, задержащую все парамеры списка показа
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = app->getSurface();                                   //поверхность в которую новый список показа будет показывать
    createInfo.minImageCount = imageCount;                          //число изображений в списке показа. Например чтобы задать двойную или тройную буфферизацию, необходимо задать это значение соответственно 2 и 3
                                                                    //Задание значения равным 1 представляет собой запрос на рендериг прямо в передний буфер или непосредственно на дисплей.
    createInfo.imageFormat = surfaceFormat.format;                  //используем ранее найденый формат и просто передаём его
    createInfo.imageColorSpace = surfaceFormat.colorSpace;          //и цветовое пространство
    createInfo.imageExtent = extent;                                //это поле задаёт размер изображения в спике показа в пикселах
    createInfo.imageArrayLayers = 1;                                //и это поле задаёт число слоёв в каждом изображении. Это может быть использовано для рендеринга в изображение со слояи и дальнейшего показа отдельных слоёв пользователю
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT ;    //это набор сандартных битов из перечисления VkImageUsageFlags, задающих как изображение будет использовано. Например если вы хотите осуществлять рендериг в изображение
                                                                    //как обычное цветовое подключение, то вам нужно включть бит VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, и если вы хотите писать в него прямо из шейдера, то выключите бит VK_IMAGE_USAGE_STORAGE_BIT
    QueueFamilyIndices indices = app->getQueueFamilyIndices();
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;   //поле задаёт как именно изображение будет совместно использоваться различными очередями, этот бит означает что изображение будет использовано несколькими очередями
        createInfo.pQueueFamilyIndices = queueFamilyIndices;        //в этом случае в данном поле задаётся указатель на массив индексов очередей, в которых эти изображения будут использоваться
        createInfo.queueFamilyIndexCount = 2;                       //и задаётся длина этого массива
    } else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;    //бит для случая, если изображение будет использоваться только одной очередью
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;   //Задаёт как изображения должны быть преобразованы перед показом пользователю. Это поле позволяетс поворачивать или переворачивать изображение для учёта таких вещей, как дисплей с портативной ориентацией
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;              //управляет тем, как смешивание с учёто альфа-канала  осуществляется оконной системой. При данном биде альфа канал изобращения игнорируется и равен 1.0.
    createInfo.presentMode = presentMode;                                       //поле управляет синхронизацией с оконной системой и скоростью, с которой изображения показываются на поверхность.
    createInfo.clipped = VK_TRUE;                                               //поле для оптимизации случая когда не вся поверхность видна. Избежание рендеринга частй которые не видит пользователь
    createInfo.oldSwapchain = VK_NULL_HANDLE;                                   //поле для передачи старого списк показа для переиспользования

    if (vkCreateSwapchainKHR(app->getDevice(), &createInfo, nullptr, &swapChain) != VK_SUCCESS)   //функция дял создания цепочки обмена, устройство с которым связан список показа передаётся в параметре device
    {                                                                                   //информация о списке показа передаётся в виде структуры VkSwapchainCreateInfoKHR которая определена выше
        throw std::runtime_error("failed to create swap chain!");
    }

    //записываем дескриптор изображений представляющий элементы в списке показа
    vkGetSwapchainImagesKHR(app->getDevice(), swapChain, &imageCount, nullptr);                   //в imageCount записываем точно число изображений

    swapChainAttachments.resize(2);
    for(size_t i=0;i<swapChainAttachments.size();i++)
    {
        swapChainAttachments.at(i).image.resize(imageCount);
        swapChainAttachments.at(i).imageView.resize(imageCount);
        swapChainAttachments.at(i).setSize(imageCount);
        vkGetSwapchainImagesKHR(app->getDevice(), swapChain, &imageCount, swapChainAttachments.at(i).image.data());    //получаем дескрипторы, на них мы будем ссылаться при рендеринге

    }

    swapChainImageFormat = surfaceFormat.format;                                        //сохраним форматы
    swapChainExtent = extent;                                                           //и размеры
}
    //Формат поверхности
    VkSurfaceFormatKHR postProcessing::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)   //ожидаем получить нелинейные sRGB - данные
            {
                return availableFormat;
            }
        }
        return availableFormats[0];     //в противном случае возвращаем первое что есть
    }
    //Режим презентации
    VkPresentModeKHR postProcessing::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)    //аналогичная процедура поиска для режима показа
            {                                                           //подробно про все биты страница 136
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    //Экстент подкачки - это разрешение изображений цепочки подкачки, и оно почти всегда точно равно разрешению окна, в которое мы рисуем, в пикселях
    VkExtent2D postProcessing::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(&app->getWindow(), &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

void postProcessing::createImageViews()
{
    /* Вид изображения позволяет части или всему изображению рассматриваться при помощи другого фрагмента.
     * Получающийся вид родительского изображения должен иметь те же размеры, что и родитель, хотя в вид
     * может входить часть элементов массива или уровней в пирамиде. Форматы родительского и дочернего
     * изображений должны быть совместимы, что обычно означает, что у них одинаковое чисо бит на пиксел,
     * даже если форматы полностью отличаются и даже если отличаетс ячисло каналова изображений*/

    for(size_t i=0;i<swapChainAttachments.size();i++)
    {
        for (size_t size = 0; size < swapChainAttachments.at(i).getSize(); size++)
        {
            swapChainAttachments.at(i).imageView[size] = createImageView(app,swapChainAttachments.at(i).image[size], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }
}


//=======================================RenderPass======================//

void postProcessing::createRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                       //следующие 4 параметра смотри на странице 210
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;              //в каком размещении его нужно оставить по завершению рендеринга

    VkAttachmentDescription bloomColorAttachment{};
    bloomColorAttachment.format = swapChainImageFormat;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
    bloomColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
    bloomColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                       //следующие 4 параметра смотри на странице 210
    bloomColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    bloomColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    bloomColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    bloomColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
    bloomColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;              //в каком размещении его нужно оставить по завершению рендеринга

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;                                          //индекс в массив подключений
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;       //размещение

    VkAttachmentReference bloomColorAttachmentRef{};
    bloomColorAttachmentRef.attachment = 1;                                          //индекс в массив подключений
    bloomColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;       //размещение

    VkAttachmentReference attachmentRef[2] = {colorAttachmentRef,bloomColorAttachmentRef};

    VkSubpassDescription subpass{};                                             //подпроходы рендеринга
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;                //бит для графики
    subpass.colorAttachmentCount = 2;                                           //количество подключений
    subpass.pColorAttachments = attachmentRef;                                 //подключения

    VkSubpassDependency dependency{};                                           //зависимости
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;                                //ссылка из исходного прохода (создавшего данные)
    dependency.dstSubpass = 0;                                                  //в целевой подпроход (поглощающий данные)
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;       //задаёт как стадии конвейера в исходном проходе создают данные
    dependency.srcAccessMask = 0;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, bloomColorAttachment};
    //информация о проходе рендеринга
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());         //количество структур VkAtachmentDescription, определяющих подключения, связанные с этим проходом рендеринга
    renderPassInfo.pAttachments = attachments.data();                                   //Каждая структура определяет одно изображение, которое будет использовано как входное, выходное или входное и выходное одновремнно для оного или нескольких проходо в данном редеринге
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)    //создаём проход рендеринга
    {
        throw std::runtime_error("failed to create render pass!");
    }
}

void postProcessing::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding2{};
    samplerLayoutBinding2.binding = 1;
    samplerLayoutBinding2.descriptorCount = 1;
    samplerLayoutBinding2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding2.pImmutableSamplers = nullptr;
    samplerLayoutBinding2.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding3{};
    samplerLayoutBinding3.binding = 2;
    samplerLayoutBinding3.descriptorCount = 1;
    samplerLayoutBinding3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding3.pImmutableSamplers = nullptr;
    samplerLayoutBinding3.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 3> textureBindings = {samplerLayoutBinding,samplerLayoutBinding2,samplerLayoutBinding3};
    VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
    textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    textureLayoutInfo.bindingCount = static_cast<uint32_t>(textureBindings.size());
    textureLayoutInfo.pBindings = textureBindings.data();

    if (vkCreateDescriptorSetLayout(app->getDevice(), &textureLayoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void postProcessing::createGraphicsPipeline()
{
    //считываем шейдеры
    auto vertShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\postProcessing\\postProcessingVert.spv");
    auto fragShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\postProcessing\\postProcessingFrag.spv");
    //создаём шейдерные модули
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    //задаём стадии шейдеров в конвейере
    //вершинный
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //фрагментный
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;                           //ниформацию о всех битах смотри на странице 222
    fragShaderStageInfo.module = fragShaderModule;                                      //сюда передаём шейдерный модуль
    fragShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    /* Преобразование области вывода - это последнее преобразование координат в конвейере Vulkan до растретизации.
     * Оно преобразует координаты вершины из нормализованных координат устройства в оконные координаты. Одновременно
     * может использоваться несколько областей вывода.*/

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    /* Растеризация - это процесс, в ходе которого примитивы, представленные вершинами, преобразуются в потоки фрагментов, которых к обработке
     * фрагментным шейдером. Состояние растеризации управляется тем, как этот процесс происходит, и задаётся при помощи следующей структуры*/

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;                                     //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_FALSE;                                              //используется для того чтобы включать отсечение глубины
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional                              //
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    /* Мультсемплинг - это процесс создания нескольких образцов (sample) для каждого пиксела в изображении.
     * Они используются для борьбы с алиансингом и может заметно улучшить общее качество изображения при эффективном использовании*/

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    VkPipelineColorBlendAttachmentState colorBlendAttachment[2];
    colorBlendAttachment[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[0].blendEnable = VK_TRUE;
    colorBlendAttachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].alphaBlendOp = VK_BLEND_OP_MAX;                         // Optional

    colorBlendAttachment[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[1].blendEnable = VK_TRUE;
    colorBlendAttachment[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].alphaBlendOp = VK_BLEND_OP_MAX;                         // Optional

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                         //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                       //Optional
    colorBlending.attachmentCount = 2;                                              //количество подключений
    colorBlending.pAttachments = colorBlendAttachment;                              //массив подключений
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
     * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
     * при помощи команд прямо внутри командного буфера*/

    VkDescriptorSetLayout SetLayouts[1] = {descriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = SetLayouts;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;                                            //число структур в массиве структур
    pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
    pipelineInfo.pVertexInputState = &vertexInputInfo;                               //вершинный ввод
    pipelineInfo.pInputAssemblyState = &inputAssembly;                             //фаза входной сборки
    pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
    pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
    pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
    pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
    pipelineInfo.layout = pipelineLayout;                              //
    pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
    pipelineInfo.subpass = 0;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

//===================Framebuffers===================================

void postProcessing::createFramebuffers()
{
    framebuffers.resize(imageCount);
    for (size_t i = 0; i < framebuffers.size(); i++)
    {
        std::array<VkImageView, 2> attachments =
        {
            swapChainAttachments.at(0).imageView[i],
            swapChainAttachments.at(1).imageView[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;                                                                    //дескриптор объекта прохода рендеринга
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                //число изображений
        framebufferInfo.pAttachments = attachments.data();                                                          //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
        framebufferInfo.width = swapChainExtent.width;                                                              //ширина изображения
        framebufferInfo.height = swapChainExtent.height;                                                            //высота изображения
        framebufferInfo.layers = 1;                                                                                 //число слоёв

        if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) //создание буфера кадров
        {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void postProcessing::createDescriptorPool(std::vector<attachments> & Attachments)
{
    std::vector<VkDescriptorPoolSize> poolSizes(3);
    size_t index = 0;

    poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes.at(index).descriptorCount = static_cast<uint32_t>(Attachments[index].imageView.size());
    index++;
    poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes.at(index).descriptorCount = static_cast<uint32_t>(Attachments[index].imageView.size());
    index++;
    poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes.at(index).descriptorCount = static_cast<uint32_t>(Attachments[index].imageView.size());
    index++;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void postProcessing::createDescriptorSets(std::vector<attachments> & Attachments)
{
    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for(size_t i=0;i<Attachments.size();i++)
    {
        VkSamplerCreateInfo bloomCenterSamplerInfo{};
        bloomCenterSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        bloomCenterSamplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
        bloomCenterSamplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
        bloomCenterSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Режим адресации
        bloomCenterSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
        bloomCenterSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Повторение текстуры при выходе за пределы размеров изображения.
        bloomCenterSamplerInfo.anisotropyEnable = VK_FALSE;
        bloomCenterSamplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
        bloomCenterSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
        bloomCenterSamplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
        bloomCenterSamplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
        bloomCenterSamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
        bloomCenterSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        bloomCenterSamplerInfo.minLod = 0.0f;
        bloomCenterSamplerInfo.maxLod = 1.0f;
        bloomCenterSamplerInfo.mipLodBias = 0.0f; // Optional

        if (vkCreateSampler(app->getDevice(), &bloomCenterSamplerInfo, nullptr, &Attachments[i].sampler) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    for (size_t image = 0; image < imageCount; image++)
    {
        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
        std::array<VkDescriptorImageInfo, 3> imageInfo;

        for(size_t i=0;i<Attachments.size();i++)
        {
            imageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[i].imageView = Attachments[i].imageView[image];
            imageInfo[i].sampler = Attachments[i].sampler;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = descriptorSets[image];
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pImageInfo = &imageInfo[i];
        }

        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}


void postProcessing::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i)
{
    std::array<VkClearValue, 2> ClearValues{};
    ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    ClearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo ectsRenderPassInfo{};
    ectsRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    ectsRenderPassInfo.renderPass = renderPass;
    ectsRenderPassInfo.framebuffer = framebuffers[i];
    ectsRenderPassInfo.renderArea.offset = {0, 0};
    ectsRenderPassInfo.renderArea.extent = swapChainExtent;
    ectsRenderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
    ectsRenderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffers[i], &ectsRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
        vkCmdDraw(commandBuffers[i], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[i]);
}

uint32_t &postProcessing::ImageCount()
{
    return imageCount;
}

VkPipeline &postProcessing::GraphicsPipeline()
{
    return graphicsPipeline;
}

VkPipelineLayout &postProcessing::PipelineLayout()
{
    return pipelineLayout;
}

VkDescriptorSetLayout &postProcessing::DescriptorSetLayout()
{
    return descriptorSetLayout;
}

VkSwapchainKHR &postProcessing::SwapChain()
{
    return swapChain;
}

VkFormat &postProcessing::SwapChainImageFormat()
{
    return swapChainImageFormat;
}

VkExtent2D &postProcessing::SwapChainExtent()
{
    return swapChainExtent;
}

VkRenderPass &postProcessing::RenderPass()
{
    return renderPass;
}

std::vector<VkFramebuffer> &postProcessing::Framebuffers()
{
    return framebuffers;
}

VkDescriptorPool &postProcessing::DescriptorPool()
{
    return descriptorPool;
}

std::vector<VkDescriptorSet> &postProcessing::DescriptorSets()
{
    return descriptorSets;
}
