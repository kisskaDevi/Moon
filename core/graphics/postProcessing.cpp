#include "graphics.h"
#include "core/operations.h"

#include <cstdint>          // нужна для UINT32_MAX
#include <array>
#include <algorithm>        // нужна для std::min/std::max

postProcessing::postProcessing(){}

void postProcessing::setApplication(VkApplication * app){this->app = app;}
void postProcessing::setMSAASamples(VkSampleCountFlagBits msaaSamples){this->msaaSamples = msaaSamples;}

void postProcessing::destroy()
{
    vkDestroyPipeline(app->getDevice(), first.Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), first.PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), first.DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), first.DescriptorPool, nullptr);

    vkDestroyPipeline(app->getDevice(), second.Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), second.PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), second.DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), second.DescriptorPool, nullptr);

    vkDestroyRenderPass(app->getDevice(), renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        vkDestroyFramebuffer(app->getDevice(), framebuffers[i],nullptr);

    for(size_t i=0; i<Attachments.size(); i++){
        Attachments.at(i).deleteAttachment(&app->getDevice());
        Attachments.at(i).deleteSampler(&app->getDevice());
    }

    for(size_t i=0; i<swapChainAttachments.size(); i++)
        for(size_t image=0; image <imageCount;image++)
            vkDestroyImageView(app->getDevice(),swapChainAttachments.at(i).imageView[image],nullptr);
    vkDestroySwapchainKHR(app->getDevice(), swapChain, nullptr);
}

void postProcessing::createAttachments(GLFWwindow* window, SwapChainSupportDetails swapChainSupport)
{
    createSwapChain(window, swapChainSupport);
    createImageViews();
    createColorAttachments();
}
    //Создание цепочки обмена
    void postProcessing::createSwapChain(GLFWwindow* window, SwapChainSupportDetails swapChainSupport)
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

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);                                   //задаём поддерживаемый формат
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);                                    //смотри ниже
        VkExtent2D extent = chooseSwapExtent(window, swapChainSupport.capabilities);                                                    //задаём размер изображения в списке показа в пикселях

        imageCount = swapChainSupport.capabilities.minImageCount + 1;                                                           //запрос на поддержку минимального количества числа изображений, число изображений равное 2 означает что один буфер передний, а второй задний
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)        //в первом условии мы проверяем доступно ли нам вообще какое-то количество изображений
        {                                                                                                                       //и проверяем не совпадает ли максимальное число изображений с минимальным
            imageCount = swapChainSupport.capabilities.maxImageCount;                                                           //присываиваем максимальное значение
        }

        QueueFamilyIndices indices = app->getQueueFamilyIndices();
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        //Создаём соответствующую структуру, задержащую все парамеры списка показа
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = app->getSurface();                         //поверхность в которую новый список показа будет показывать
        createInfo.minImageCount = imageCount;                          //число изображений в списке показа. Например чтобы задать двойную или тройную буфферизацию, необходимо задать это значение соответственно 2 и 3
                                                                        //Задание значения равным 1 представляет собой запрос на рендериг прямо в передний буфер или непосредственно на дисплей.
        createInfo.imageFormat = surfaceFormat.format;                  //используем ранее найденый формат и просто передаём его
        createInfo.imageColorSpace = surfaceFormat.colorSpace;          //и цветовое пространство
        createInfo.imageExtent = extent;                                //это поле задаёт размер изображения в спике показа в пикселах
        createInfo.imageArrayLayers = 1;                                //и это поле задаёт число слоёв в каждом изображении. Это может быть использовано для рендеринга в изображение со слояи и дальнейшего показа отдельных слоёв пользователю
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ;    //это набор сандартных битов из перечисления VkImageUsageFlags, задающих как изображение будет использовано. Например если вы хотите осуществлять рендериг в изображение
                                                                        //как обычное цветовое подключение, то вам нужно включть бит VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, и если вы хотите писать в него прямо из шейдера, то выключите бит VK_IMAGE_USAGE_STORAGE_BIT
        if(indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;   //поле задаёт как именно изображение будет совместно использоваться различными очередями, этот бит означает что изображение будет использовано несколькими очередями
            createInfo.pQueueFamilyIndices = queueFamilyIndices;        //в этом случае в данном поле задаётся указатель на массив индексов очередей, в которых эти изображения будут использоваться
            createInfo.queueFamilyIndexCount = 2;                       //и задаётся длина этого массива
        }else{
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;    //бит для случая, если изображение будет использоваться только одной очередью
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;   //Задаёт как изображения должны быть преобразованы перед показом пользователю. Это поле позволяетс поворачивать или переворачивать изображение для учёта таких вещей, как дисплей с портативной ориентацией
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;              //управляет тем, как смешивание с учёто альфа-канала  осуществляется оконной системой. При данном биде альфа канал изобращения игнорируется и равен 1.0.
        createInfo.presentMode = presentMode;                                       //поле управляет синхронизацией с оконной системой и скоростью, с которой изображения показываются на поверхность.
        createInfo.clipped = VK_TRUE;                                               //поле для оптимизации случая когда не вся поверхность видна. Избежание рендеринга частй которые не видит пользователь
        createInfo.oldSwapchain = VK_NULL_HANDLE;                                   //поле для передачи старого списк показа для переиспользования

        if (vkCreateSwapchainKHR(app->getDevice(), &createInfo, nullptr, &swapChain) != VK_SUCCESS)     //функция дял создания цепочки обмена, устройство с которым связан список показа передаётся в параметре device
            throw std::runtime_error("failed to create swap chain!");                                   //информация о списке показа передаётся в виде структуры VkSwapchainCreateInfoKHR которая определена выше

        vkGetSwapchainImagesKHR(app->getDevice(), swapChain, &imageCount, nullptr);                     //записываем дескриптор изображений представляющий элементы в списке показа

        swapChainAttachments.resize(swapChainAttachmentCount);
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
        //Экстент обмена - это разрешение изображений цепочки обмена, и оно почти всегда точно равно разрешению окна, в которое мы рисуем, в пикселях
        VkExtent2D postProcessing::chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities)
        {
            if (capabilities.currentExtent.width != UINT32_MAX)
                return capabilities.currentExtent;
            else
            {
                int width, height;
                glfwGetFramebufferSize(window, &width, &height);

                VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

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
            for (size_t size = 0; size < swapChainAttachments.at(i).getSize(); size++)
                swapChainAttachments.at(i).imageView[size] = createImageView(app,swapChainAttachments.at(i).image[size], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    void postProcessing::createColorAttachments()
    {
        Attachments.resize(AttachmentCount);
        for(size_t i=0;i<AttachmentCount;i++)
        {
            Attachments[i].resize(imageCount);
            for(size_t image=0; image<imageCount; image++)
            {
                createImage(app,swapChainExtent.width,swapChainExtent.height,1,VK_SAMPLE_COUNT_1_BIT,swapChainImageFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
                Attachments[i].imageView[image] = createImageView(app, Attachments[i].image[image], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
            }
        }
        for(size_t i=0;i<Attachments.size();i++)
        {
            VkSamplerCreateInfo SamplerInfo{};
                SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                SamplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
                SamplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
                SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Режим адресации
                SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
                SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Повторение текстуры при выходе за пределы размеров изображения.
                SamplerInfo.anisotropyEnable = VK_TRUE;
                SamplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
                SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
                SamplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
                SamplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
                SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
                SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                SamplerInfo.minLod = 0.0f;
                SamplerInfo.maxLod = 0.0f;
                SamplerInfo.mipLodBias = 0.0f;

            if (vkCreateSampler(app->getDevice(), &SamplerInfo, nullptr, &Attachments[i].sampler) != VK_SUCCESS)
                throw std::runtime_error("failed to create postProcessing sampler!");
        }
    }

//=======================================RenderPass======================//

void postProcessing::createRenderPass()
{
    uint32_t index = 0;

    std::array<VkAttachmentDescription,2> attachments{};
        attachments[index].format = swapChainImageFormat;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                       //следующие 4 параметра смотри на странице 210
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;              //в каком размещении его нужно оставить по завершению рендеринга
    index++;
        attachments[index].format = swapChainImageFormat;
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    index = 0;
    std::array<VkAttachmentReference,2> firstAttachmentRef;
        firstAttachmentRef[index].attachment = 0;
        firstAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    index++;
        firstAttachmentRef[index].attachment = 1;
        firstAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    index = 0;
    std::array<VkAttachmentReference,1> secondAttachmentRef;
        secondAttachmentRef[index].attachment = 0;
        secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    index = 0;
    std::array<VkAttachmentReference,1> inSecondAttachmentRef;
        inSecondAttachmentRef[index].attachment = 1;
        inSecondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    index = 0;
    std::array<VkAttachmentReference,1> thirdAttachmentRef;
        thirdAttachmentRef[index].attachment = 0;
        thirdAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    index = 0;
    std::array<VkSubpassDescription,3> subpass{};
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(firstAttachmentRef.size());
        subpass[index].pColorAttachments = firstAttachmentRef.data();
    index++;
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(secondAttachmentRef.size());
        subpass[index].pColorAttachments = secondAttachmentRef.data();
        subpass[index].inputAttachmentCount = static_cast<uint32_t>(inSecondAttachmentRef.size());
        subpass[index].pInputAttachments = inSecondAttachmentRef.data();
    index++;
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(thirdAttachmentRef.size());
        subpass[index].pColorAttachments = thirdAttachmentRef.data();

    index = 0;
    std::array<VkSubpassDependency,3> dependency{};                                                                                        //зависимости
        dependency[index].srcSubpass = VK_SUBPASS_EXTERNAL;                                                                                //ссылка из исходного прохода (создавшего данные)
        dependency[index].dstSubpass = 0;                                                                                                  //в целевой подпроход (поглощающий данные)
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;                                                                 //задаёт как стадии конвейера в исходном проходе создают данные
        dependency[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    index++;
        dependency[index].srcSubpass = 0;
        dependency[index].dstSubpass = 1;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency[index].srcAccessMask = 0;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency[index].dstAccessMask = 0;
    index++;
        dependency[index].srcSubpass = 1;
        dependency[index].dstSubpass = 2;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].srcAccessMask = 0;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependency.size());
        renderPassInfo.pDependencies = dependency.data();

    if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing render pass!");
}

//===================Framebuffers===================================

void postProcessing::createFramebuffers()
{
    framebuffers.resize(imageCount);
    for (size_t i = 0; i < framebuffers.size(); i++)
    {
        uint32_t index = 0;

        std::vector<VkImageView> attachments(swapChainAttachments.size()+Attachments.size());
            for(size_t j=0;j<swapChainAttachments.size();j++){
                attachments.at(index) = swapChainAttachments.at(j).imageView[i]; index++;}
            for(size_t j=0;j<Attachments.size();j++){
                attachments.at(index) = Attachments.at(j).imageView[i]; index++;}

        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;                                                                    //дескриптор объекта прохода рендеринга
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                //число изображений
            framebufferInfo.pAttachments = attachments.data();                                                          //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
            framebufferInfo.width = swapChainExtent.width;                                                              //ширина изображения
            framebufferInfo.height = swapChainExtent.height;                                                            //высота изображения
            framebufferInfo.layers = 1;                                                                                 //число слоёв
        if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) //создание буфера кадров
            throw std::runtime_error("failed to create postProcessing framebuffer!");
    }
}

//===================Pipelines===================================

void postProcessing::createPipelines()
{
    createDescriptorSetLayout();
    createFirstGraphicsPipeline();
    createSecondGraphicsPipeline();
}
    void postProcessing::createDescriptorSetLayout()
    {
        uint32_t index = 0;

        std::array<VkDescriptorSetLayoutBinding,1> firstBindings{};
            firstBindings[index].binding = 0;
            firstBindings[index].descriptorCount = 1;
            firstBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            firstBindings[index].pImmutableSamplers = nullptr;
            firstBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        index = 0;
        std::array<VkDescriptorSetLayoutBinding,5> secondBindings{};
            secondBindings[index].binding = index;
            secondBindings[index].descriptorCount = 1;
            secondBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings[index].pImmutableSamplers = nullptr;
            secondBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            secondBindings[index].binding = index;
            secondBindings[index].descriptorCount = 1;
            secondBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings[index].pImmutableSamplers = nullptr;
            secondBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            secondBindings[index].binding = index;
            secondBindings[index].descriptorCount = 1;
            secondBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings[index].pImmutableSamplers = nullptr;
            secondBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            secondBindings[index].binding = index;
            secondBindings[index].descriptorCount = 1;
            secondBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings[index].pImmutableSamplers = nullptr;
            secondBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            secondBindings[index].binding = index;
            secondBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            secondBindings[index].descriptorCount = 1;
            secondBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            secondBindings[index].pImmutableSamplers = nullptr;

        index = 0;
        std::array<VkDescriptorSetLayoutCreateInfo,2> textureLayoutInfo{};
            textureLayoutInfo[index].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo[index].bindingCount = static_cast<uint32_t>(firstBindings.size());
            textureLayoutInfo[index].pBindings = firstBindings.data();
        index++;
            textureLayoutInfo[index].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo[index].bindingCount = static_cast<uint32_t>(secondBindings.size());
            textureLayoutInfo[index].pBindings = secondBindings.data();

        if (vkCreateDescriptorSetLayout(app->getDevice(), &textureLayoutInfo.at(0), nullptr, &first.DescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing descriptor set layout 1!");
        if (vkCreateDescriptorSetLayout(app->getDevice(), &textureLayoutInfo.at(1), nullptr, &second.DescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing descriptor set layout 2!");
    }
    void postProcessing::createFirstGraphicsPipeline()
    {
        uint32_t index = 0;

        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\firstPostProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\firstPostProcessingFrag.spv");
        VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
        std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
            shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
            shaderStages[index].module = vertShaderModule;
            shaderStages[index].pName = "main";
        index++;
            shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            shaderStages[index].module = fragShaderModule;
            shaderStages[index].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount = 0;
            vertexInputInfo.pVertexBindingDescriptions = nullptr;
            vertexInputInfo.vertexAttributeDescriptionCount = 0;
            vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

        index = 0;
        std::array<VkViewport,1> viewport{};
            viewport[index].x = 0.0f;
            viewport[index].y = 0.0f;
            viewport[index].width  = (float) swapChainExtent.width;
            viewport[index].height= (float) swapChainExtent.height;
            viewport[index].minDepth = 0.0f;
            viewport[index].maxDepth = 1.0f;
        std::array<VkRect2D,1> scissor{};
            scissor[index].offset = {0, 0};
            scissor[index].extent = swapChainExtent;
        VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = static_cast<uint32_t>(viewport.size());;
            viewportState.pViewports = viewport.data();
            viewportState.scissorCount = static_cast<uint32_t>(scissor.size());;
            viewportState.pScissors = scissor.data();

        VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasClamp = 0.0f;
            rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

        index = 0;
        std::array<VkPipelineColorBlendAttachmentState,2> colorBlendAttachment;
            colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment[index].blendEnable = VK_FALSE;
            colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
            colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
        index++;
            colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment[index].blendEnable = VK_FALSE;
            colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
            colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
        VkPipelineColorBlendStateCreateInfo colorBlending{};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = VK_LOGIC_OP_COPY;
            colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());
            colorBlending.pAttachments = colorBlendAttachment.data();
            colorBlending.blendConstants[0] = 0.0f;
            colorBlending.blendConstants[1] = 0.0f;
            colorBlending.blendConstants[2] = 0.0f;
            colorBlending.blendConstants[3] = 0.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_FALSE;
            depthStencil.depthWriteEnable = VK_FALSE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.minDepthBounds = 0.0f;
            depthStencil.maxDepthBounds = 1.0f;
            depthStencil.stencilTestEnable = VK_FALSE;
            depthStencil.front = {};
            depthStencil.back = {};

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &first.DescriptorSetLayout;
        if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &first.PipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing pipeline layout 1!");

        VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
            pipelineInfo.pStages = shaderStages.data();
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.layout = first.PipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo.pDepthStencilState = &depthStencil;
        if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &first.Pipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing graphics pipeline 1!");

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
    }
    void postProcessing::createSecondGraphicsPipeline()
    {
        uint32_t index = 0;

        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\postProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\postProcessingFrag.spv");
        VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
        std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
            shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
            shaderStages[index].module = vertShaderModule;
            shaderStages[index].pName = "main";
        index++;
            shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            shaderStages[index].module = fragShaderModule;
            shaderStages[index].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount = 0;
            vertexInputInfo.pVertexBindingDescriptions = nullptr;
            vertexInputInfo.vertexAttributeDescriptionCount = 0;
            vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

        index = 0;
        std::array<VkViewport,1> viewport{};
            viewport[index].x = 0.0f;
            viewport[index].y = 0.0f;
            viewport[index].width  = (float) swapChainExtent.width;
            viewport[index].height= (float) swapChainExtent.height;
            viewport[index].minDepth = 0.0f;
            viewport[index].maxDepth = 1.0f;
        std::array<VkRect2D,1> scissor{};
            scissor[index].offset = {0, 0};
            scissor[index].extent = swapChainExtent;
        VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = static_cast<uint32_t>(viewport.size());;
            viewportState.pViewports = viewport.data();
            viewportState.scissorCount = static_cast<uint32_t>(scissor.size());;
            viewportState.pScissors = scissor.data();

        VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasClamp = 0.0f;
            rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

        index = 0;
        std::array<VkPipelineColorBlendAttachmentState,1> colorBlendAttachment;
            colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment[index].blendEnable = VK_FALSE;
            colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
            colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
        VkPipelineColorBlendStateCreateInfo colorBlending{};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = VK_LOGIC_OP_COPY;
            colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());
            colorBlending.pAttachments = colorBlendAttachment.data();
            colorBlending.blendConstants[0] = 0.0f;
            colorBlending.blendConstants[1] = 0.0f;
            colorBlending.blendConstants[2] = 0.0f;
            colorBlending.blendConstants[3] = 0.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_FALSE;
            depthStencil.depthWriteEnable = VK_FALSE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.minDepthBounds = 0.0f;
            depthStencil.maxDepthBounds = 1.0f;
            depthStencil.stencilTestEnable = VK_FALSE;
            depthStencil.front = {};
            depthStencil.back = {};

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &second.DescriptorSetLayout;
        if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &second.PipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing pipeline layout 2!");

        index = 0;
        std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfo{};
            pipelineInfo[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo[index].stageCount = static_cast<uint32_t>(shaderStages.size());
            pipelineInfo[index].pStages = shaderStages.data();
            pipelineInfo[index].pVertexInputState = &vertexInputInfo;
            pipelineInfo[index].pInputAssemblyState = &inputAssembly;
            pipelineInfo[index].pViewportState = &viewportState;
            pipelineInfo[index].pRasterizationState = &rasterizer;
            pipelineInfo[index].pMultisampleState = &multisampling;
            pipelineInfo[index].pColorBlendState = &colorBlending;
            pipelineInfo[index].layout = second.PipelineLayout;
            pipelineInfo[index].renderPass = renderPass;
            pipelineInfo[index].subpass = 2;
            pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo[index].pDepthStencilState = &depthStencil;
        if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &second.Pipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing graphics pipeline 2!");

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
    }

void postProcessing::createDescriptorPool()
{
    size_t index = 0;
    std::array<VkDescriptorPoolSize,1> firstPoolSizes;
        firstPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        firstPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    VkDescriptorPoolCreateInfo firstPoolInfo{};
        firstPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        firstPoolInfo.poolSizeCount = static_cast<uint32_t>(firstPoolSizes.size());
        firstPoolInfo.pPoolSizes = firstPoolSizes.data();
        firstPoolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(app->getDevice(), &firstPoolInfo, nullptr, &first.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing descriptor pool 1!");

    index = 0;
    std::array<VkDescriptorPoolSize,5> secondPoolSizes;
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    VkDescriptorPoolCreateInfo secondPoolInfo{};
        secondPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        secondPoolInfo.poolSizeCount = static_cast<uint32_t>(secondPoolSizes.size());
        secondPoolInfo.pPoolSizes = secondPoolSizes.data();
        secondPoolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(app->getDevice(), &secondPoolInfo, nullptr, &second.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing descriptor pool 2!");
}

void postProcessing::createDescriptorSets(std::vector<attachments> & Attachments, std::vector<VkBuffer>& uniformBuffers)
{
    first.DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> firstLayouts(imageCount, first.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo firstAllocInfo{};
        firstAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        firstAllocInfo.descriptorPool = first.DescriptorPool;
        firstAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        firstAllocInfo.pSetLayouts = firstLayouts.data();
    if (vkAllocateDescriptorSets(app->getDevice(), &firstAllocInfo, first.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate postProcessing descriptor sets 1!");

    for (size_t image = 0; image < imageCount; image++)
    {
        uint32_t index = 0;

        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
        std::array<VkDescriptorImageInfo, 1> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments[1].imageView[image];
            imageInfo[index].sampler = Attachments[1].sampler;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = first.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    second.DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, second.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = second.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, second.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate postProcessing descriptor sets 2!");

    for (size_t image = 0; image < imageCount; image++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorImageInfo, 5> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments[0].imageView[image];
            imageInfo[index].sampler = Attachments[0].sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = this->Attachments[0].imageView[image];
            imageInfo[index].sampler = this->Attachments[0].sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments[2].imageView[image];
            imageInfo[index].sampler = Attachments[2].sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments[3].imageView[image];
            imageInfo[index].sampler = Attachments[3].sampler;

        VkDescriptorBufferInfo bufferInfo;
            bufferInfo.buffer = uniformBuffers[image];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

        index = 0;
        std::array<VkWriteDescriptorSet, 5> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = second.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = second.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = second.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = second.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets[image];
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void postProcessing::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i)
{
    std::array<VkClearValue, 2> ClearValues{};
        ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        ClearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, first.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, first.PipelineLayout, 0, 1, &first.DescriptorSets[i], 0, nullptr);
        vkCmdDraw(commandBuffers[i], 6, 1, 0, 0);

    vkCmdNextSubpass(commandBuffers[i], VK_SUBPASS_CONTENTS_INLINE);
    vkCmdNextSubpass(commandBuffers[i], VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, second.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, second.PipelineLayout, 0, 1, &second.DescriptorSets[i], 0, nullptr);
        vkCmdDraw(commandBuffers[i], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[i]);
}

VkSwapchainKHR                  &postProcessing::SwapChain(){return swapChain;}
uint32_t                        &postProcessing::SwapChainImageCount(){return imageCount;}
VkFormat                        &postProcessing::SwapChainImageFormat(){return swapChainImageFormat;}
VkExtent2D                      &postProcessing::SwapChainImageExtent(){return swapChainExtent;}
