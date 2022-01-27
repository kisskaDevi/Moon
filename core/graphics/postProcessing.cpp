#include "graphics.h"
#include "core/operations.h"

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

void postProcessing::createAttachments(SwapChainSupportDetails swapChainSupport)
{
    createSwapChain(swapChainSupport);
    createImageViews();
    createColorAttachments();
}
    //Создание цепочки обмена
    void postProcessing::createSwapChain(SwapChainSupportDetails swapChainSupport)
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
        createInfo.imageUsage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT ;    //это набор сандартных битов из перечисления VkImageUsageFlags, задающих как изображение будет использовано. Например если вы хотите осуществлять рендериг в изображение
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
            {throw std::runtime_error("failed to create texture sampler!");}
        }
    }

//=======================================RenderPass======================//

void postProcessing::createRenderPass()
{
    uint32_t index = 0;
    
    std::vector<VkAttachmentDescription> attachments(2);
        attachments.at(index).format = swapChainImageFormat;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
        attachments.at(index).samples = VK_SAMPLE_COUNT_1_BIT;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
        attachments.at(index).loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                       //следующие 4 параметра смотри на странице 210
        attachments.at(index).storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments.at(index).stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments.at(index).stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments.at(index).initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
        attachments.at(index).finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;              //в каком размещении его нужно оставить по завершению рендеринга
    index++;
        attachments.at(index).format = swapChainImageFormat;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
        attachments.at(index).samples = VK_SAMPLE_COUNT_1_BIT;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
        attachments.at(index).loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                       //следующие 4 параметра смотри на странице 210
        attachments.at(index).storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments.at(index).stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments.at(index).stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments.at(index).initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
        attachments.at(index).finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;     //в каком размещении его нужно оставить по завершению рендеринга
    
    index = 0;
    std::vector<VkAttachmentReference> firstAttachmentRef(2);
        firstAttachmentRef.at(index).attachment = 0;
        firstAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    index++;
        firstAttachmentRef.at(index).attachment = 1;
        firstAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
     
    index = 0;
    std::vector<VkAttachmentReference> secondAttachmentRef(1);
        secondAttachmentRef.at(index).attachment = 0;                                                  //индекс в массив подключений
        secondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;               //размещение

    index = 0;
    std::vector<VkAttachmentReference> inSecondAttachmentRef(1);
        inSecondAttachmentRef.at(index).attachment = 1;                                                  //индекс в массив подключений
        inSecondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;               //размещение

    index = 0;
    std::vector<VkAttachmentReference> thirdAttachmentRef(1);
        thirdAttachmentRef.at(index).attachment = 0;                                                  //индекс в массив подключений
        thirdAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;               //размещение
 
    index = 0;
    std::vector<VkSubpassDescription> subpass(3);                                                       //подпроходы рендеринга
        subpass.at(index).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;                          //бит для графики
        subpass.at(index).colorAttachmentCount = static_cast<uint32_t>(firstAttachmentRef.size());      //количество подключений
        subpass.at(index).pColorAttachments = firstAttachmentRef.data();                                //подключения
    index++;
        subpass.at(index).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;                          //бит для графики
        subpass.at(index).colorAttachmentCount = static_cast<uint32_t>(secondAttachmentRef.size());     //количество подключений
        subpass.at(index).pColorAttachments = secondAttachmentRef.data();                               //подключения
        subpass.at(index).inputAttachmentCount = static_cast<uint32_t>(inSecondAttachmentRef.size());
        subpass.at(index).pInputAttachments = inSecondAttachmentRef.data();
    index++;
        subpass.at(index).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;                          //бит для графики
        subpass.at(index).colorAttachmentCount = static_cast<uint32_t>(thirdAttachmentRef.size());     //количество подключений
        subpass.at(index).pColorAttachments = thirdAttachmentRef.data();                               //подключения;

    index = 0;
    std::vector<VkSubpassDependency> dependency(3);                                                                                           //зависимости
        dependency.at(index).srcSubpass = VK_SUBPASS_EXTERNAL;                                                                                //ссылка из исходного прохода (создавшего данные)
        dependency.at(index).dstSubpass = 0;                                                                                                  //в целевой подпроход (поглощающий данные)
        dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;                                                    //задаёт как стадии конвейера в исходном проходе создают данные
        dependency.at(index).srcAccessMask = 0;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
        dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    index++;
        dependency.at(index).srcSubpass = 0;                                                                                                    //ссылка из исходного прохода (создавшего данные)
        dependency.at(index).dstSubpass = 1;                                                                                                    //в целевой подпроход (поглощающий данные)
        dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;                                                                                                  //задаёт как стадии конвейера в исходном проходе создают данные
        dependency.at(index).srcAccessMask = 0;                                                                                                 //поля задают как каждый из исходных проходов обращается к данным
        dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.at(index).dstAccessMask = 0;
    index++;
        dependency.at(index).srcSubpass = 1;                                                                                                    //ссылка из исходного прохода (создавшего данные)
        dependency.at(index).dstSubpass = 2;                                                                                                    //в целевой подпроход (поглощающий данные)
        dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;                                                      //задаёт как стадии конвейера в исходном проходе создают данные
        dependency.at(index).srcAccessMask = 0;                                                                                                 //поля задают как каждый из исходных проходов обращается к данным
        dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    //информация о проходе рендеринга
    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());         //количество структур VkAtachmentDescription, определяющих подключения, связанные с этим проходом рендеринга
        renderPassInfo.pAttachments = attachments.data();                                   //Каждая структура определяет одно изображение, которое будет использовано как входное, выходное или входное и выходное одновремнно для оного или нескольких проходо в данном редеринге
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependency.size());
        renderPassInfo.pDependencies = dependency.data();

    if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)    //создаём проход рендеринга
    {
        throw std::runtime_error("failed to create render pass!");
    }
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
        {
            throw std::runtime_error("failed to create framebuffer!");
        }
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

        std::vector<VkDescriptorSetLayoutBinding> firstBindings(1);
            firstBindings.at(index).binding = 0;
            firstBindings.at(index).descriptorCount = 1;
            firstBindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            firstBindings.at(index).pImmutableSamplers = nullptr;
            firstBindings.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        index = 0;
        std::vector<VkDescriptorSetLayoutBinding> secondBindings(3);
            secondBindings.at(index).binding = 0;
            secondBindings.at(index).descriptorCount = 1;
            secondBindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings.at(index).pImmutableSamplers = nullptr;
            secondBindings.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            secondBindings.at(index).binding = 1;
            secondBindings.at(index).descriptorCount = 1;
            secondBindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings.at(index).pImmutableSamplers = nullptr;
            secondBindings.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            secondBindings.at(index).binding = 2;
            secondBindings.at(index).descriptorCount = 1;
            secondBindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings.at(index).pImmutableSamplers = nullptr;
            secondBindings.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        index = 0;
        std::vector<VkDescriptorSetLayoutCreateInfo> textureLayoutInfo(2);
            textureLayoutInfo.at(index).sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo.at(index).bindingCount = static_cast<uint32_t>(firstBindings.size());
            textureLayoutInfo.at(index).pBindings = firstBindings.data();
        index++;
            textureLayoutInfo.at(index).sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo.at(index).bindingCount = static_cast<uint32_t>(secondBindings.size());
            textureLayoutInfo.at(index).pBindings = secondBindings.data();

        if (vkCreateDescriptorSetLayout(app->getDevice(), &textureLayoutInfo.at(0), nullptr, &first.DescriptorSetLayout) != VK_SUCCESS)
        {throw std::runtime_error("failed to create descriptor set layout 1!");}
        if (vkCreateDescriptorSetLayout(app->getDevice(), &textureLayoutInfo.at(1), nullptr, &second.DescriptorSetLayout) != VK_SUCCESS)
        {throw std::runtime_error("failed to create descriptor set layout 2!");}
    }
    void postProcessing::createFirstGraphicsPipeline()
    {
        //считываем шейдеры
        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\firstPostProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\firstPostProcessingFrag.spv");
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

        std::array<VkPipelineColorBlendAttachmentState,2> colorBlendAttachment;
        colorBlendAttachment[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[0].blendEnable = VK_FALSE;
        colorBlendAttachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].alphaBlendOp = VK_BLEND_OP_MAX;

        colorBlendAttachment[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[1].blendEnable = VK_FALSE;
        colorBlendAttachment[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[1].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[1].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[1].alphaBlendOp = VK_BLEND_OP_MAX;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;                                                 //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
        colorBlending.logicOp = VK_LOGIC_OP_COPY;                                               //Optional
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());     //количество подключений
        colorBlending.pAttachments = colorBlendAttachment.data();                               //массив подключений
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
         * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
         * при помощи команд прямо внутри командного буфера*/

        VkDescriptorSetLayout SetLayouts[1] = {first.DescriptorSetLayout};
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

        if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &first.PipelineLayout) != VK_SUCCESS)
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
        pipelineInfo.layout = first.PipelineLayout;                              //
        pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
        pipelineInfo.subpass = 0;                                               //подпроход рендеригка
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;

        if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &first.Pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
    }
    void postProcessing::createSecondGraphicsPipeline()
    {
        //считываем шейдеры
        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\postProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\postProcessingFrag.spv");
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

        std::array<VkPipelineColorBlendAttachmentState,1> colorBlendAttachment;
        colorBlendAttachment[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[0].blendEnable = VK_FALSE;
        colorBlendAttachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[0].alphaBlendOp = VK_BLEND_OP_MAX;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;                                                 //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
        colorBlending.logicOp = VK_LOGIC_OP_COPY;                                               //Optional
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());     //количество подключений
        colorBlending.pAttachments = colorBlendAttachment.data();                               //массив подключений
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
         * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
         * при помощи команд прямо внутри командного буфера*/

        VkDescriptorSetLayout SetLayouts[1] = {second.DescriptorSetLayout};
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

        if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &second.PipelineLayout) != VK_SUCCESS)
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
        pipelineInfo.layout = second.PipelineLayout;                              //
        pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
        pipelineInfo.subpass = 2;                                               //подпроход рендеригка
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;

        if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &second.Pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
    }

void postProcessing::createDescriptorPool()
{
    size_t index = 0;

    std::vector<VkDescriptorPoolSize> firstPoolSizes(1);
        firstPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        firstPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);

    VkDescriptorPoolCreateInfo firstPoolInfo{};
        firstPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        firstPoolInfo.poolSizeCount = static_cast<uint32_t>(firstPoolSizes.size());
        firstPoolInfo.pPoolSizes = firstPoolSizes.data();
        firstPoolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(app->getDevice(), &firstPoolInfo, nullptr, &first.DescriptorPool) != VK_SUCCESS)
    {throw std::runtime_error("failed to create descriptor pool 1!");}

    index = 0;
    std::vector<VkDescriptorPoolSize> secondPoolSizes(3);
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        secondPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);

    VkDescriptorPoolCreateInfo secondPoolInfo{};
        secondPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        secondPoolInfo.poolSizeCount = static_cast<uint32_t>(secondPoolSizes.size());
        secondPoolInfo.pPoolSizes = secondPoolSizes.data();
        secondPoolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(app->getDevice(), &secondPoolInfo, nullptr, &second.DescriptorPool) != VK_SUCCESS)
    {throw std::runtime_error("failed to create descriptor pool 2!");}
}

void postProcessing::createDescriptorSets(std::vector<attachments> & Attachments)
{
    std::vector<VkDescriptorSetLayout> firstLayouts(imageCount, first.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo firstAllocInfo{};
        firstAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        firstAllocInfo.descriptorPool = first.DescriptorPool;
        firstAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        firstAllocInfo.pSetLayouts = firstLayouts.data();

    first.DescriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &firstAllocInfo, first.DescriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets 1!");
    }

    for (size_t image = 0; image < imageCount; image++)
    {
        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
        std::array<VkDescriptorImageInfo, 1> imageInfo;

            imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[0].imageView = Attachments[1].imageView[image];
            imageInfo[0].sampler = Attachments[1].sampler;

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = first.DescriptorSets[image];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &imageInfo[0];

        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    std::vector<VkDescriptorSetLayout> layouts(imageCount, second.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = second.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();

    second.DescriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, second.DescriptorSets.data()) != VK_SUCCESS)
    {throw std::runtime_error("failed to allocate descriptor sets 2!");}

    for (size_t image = 0; image < imageCount; image++)
    {
        uint32_t i = 0;
        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
        std::array<VkDescriptorImageInfo, 3> imageInfo;

            imageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[i].imageView = Attachments[0].imageView[image];
            imageInfo[i].sampler = Attachments[0].sampler;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = second.DescriptorSets[image];
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pImageInfo = &imageInfo[i];
    i++;
            imageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[i].imageView = this->Attachments[0].imageView[image];
            imageInfo[i].sampler = this->Attachments[0].sampler;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = second.DescriptorSets[image];
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pImageInfo = &imageInfo[i];
     i++;
            imageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[i].imageView = Attachments[2].imageView[image];
            imageInfo[i].sampler = Attachments[2].sampler;

            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = second.DescriptorSets[image];
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pImageInfo = &imageInfo[i];

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

uint32_t                        &postProcessing::ImageCount(){return imageCount;}
VkSwapchainKHR                  &postProcessing::SwapChain(){return swapChain;}
VkFormat                        &postProcessing::SwapChainImageFormat(){return swapChainImageFormat;}
VkExtent2D                      &postProcessing::SwapChainExtent(){return swapChainExtent;}
