#include "postProcessing.h"

#include <cstdint>          // нужна для UINT32_MAX
#include <array>
#include <algorithm>        // нужна для std::min/std::max

postProcessing::postProcessing(){}

void postProcessing::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, QueueFamilyIndices* queueFamilyIndices, VkSurfaceKHR* surface)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
    this->queueFamilyIndices = queueFamilyIndices;
    this->surface = surface;
}
void postProcessing::setImageProp(imageInfo* pInfo)             {this->image = *pInfo;}

void  postProcessing::setBlitFactor(const float& blitFactor)    {this->blitFactor = blitFactor;}
float postProcessing::getBlitFactor()                           {return blitFactor;}

void postProcessing::destroy()
{
    vkDestroyPipeline(*device, first.Pipeline, nullptr);
    vkDestroyPipelineLayout(*device, first.PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(*device, first.DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(*device, first.DescriptorPool, nullptr);

    vkDestroyPipeline(*device, second.Pipeline, nullptr);
    vkDestroyPipelineLayout(*device, second.PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(*device, second.DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(*device, second.DescriptorPool, nullptr);

    vkDestroyRenderPass(*device, renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        vkDestroyFramebuffer(*device, framebuffers[i],nullptr);

    for(size_t i=0; i<Attachments.size(); i++){
        Attachments[i].deleteAttachment(&*device);
        Attachments[i].deleteSampler(&*device);
    }

    for(size_t i=0; i<blitAttachments.size(); i++){
        blitAttachments[i].deleteAttachment(&*device);
        blitAttachments[i].deleteSampler(&*device);
    }
    blitAttachment.deleteAttachment(&*device);
    blitAttachment.deleteSampler(&*device);

    sslrAttachment.deleteAttachment(&*device);
    sslrAttachment.deleteSampler(&*device);

    ssaoAttachment.deleteAttachment(&*device);
    ssaoAttachment.deleteSampler(&*device);

    uint32_t imageCount = image.Count;
    for(size_t i=0; i<swapChainAttachments.size(); i++)
        for(size_t image=0; image <imageCount;image++)
            vkDestroyImageView(*device,swapChainAttachments[i].imageView[image],nullptr);
    vkDestroySwapchainKHR(*device, swapChain, nullptr);
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

        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkExtent2D extent = chooseSwapExtent(window, swapChainSupport.capabilities);
        uint32_t imageCount = image.Count;

        QueueFamilyIndices indices = *queueFamilyIndices;
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        //Создаём соответствующую структуру, задержащую все парамеры списка показа
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = *surface;                                  //поверхность в которую новый список показа будет показывать
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

        if (vkCreateSwapchainKHR(*device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)     //функция дял создания цепочки обмена, устройство с которым связан список показа передаётся в параметре device
            throw std::runtime_error("failed to create swap chain!");                                   //информация о списке показа передаётся в виде структуры VkSwapchainCreateInfoKHR которая определена выше

        vkGetSwapchainImagesKHR(*device, swapChain, &imageCount, nullptr);                     //записываем дескриптор изображений представляющий элементы в списке показа

        swapChainAttachments.resize(swapChainAttachmentCount);
        for(size_t i=0;i<swapChainAttachments.size();i++)
        {
            swapChainAttachments[i].image.resize(imageCount);
            swapChainAttachments[i].imageView.resize(imageCount);
            swapChainAttachments[i].setSize(imageCount);
            vkGetSwapchainImagesKHR(*device, swapChain, &imageCount, swapChainAttachments[i].image.data());    //получаем дескрипторы, на них мы будем ссылаться при рендеринге
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
            for (size_t size = 0; size < swapChainAttachments[i].getSize(); size++)
                swapChainAttachments[i].imageView[size] =
                createImageView(    device,
                                    swapChainAttachments[i].image[size],
                                    image.Format,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    1);
    }

    void postProcessing::createColorAttachments()
    {
        uint32_t imageCount = image.Count;
        Attachments.resize(AttachmentCount);
        for(size_t i=0;i<AttachmentCount;i++)
        {
            Attachments[i].resize(image.Count);
            for(size_t imageNumber=0; imageNumber<imageCount; imageNumber++)
            {
                createImage(        physicalDevice,
                                    device,
                                    image.Extent.width,
                                    image.Extent.height,
                                    1,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    image.Format,
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    Attachments[i].image[imageNumber],
                                    Attachments[i].imageMemory[imageNumber]);

                Attachments[i].imageView[imageNumber] =
                createImageView(    device,
                                    Attachments[i].image[imageNumber],
                                    image.Format,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    1);
            }
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
            if (vkCreateSampler(*device, &SamplerInfo, nullptr, &Attachments[i].sampler) != VK_SUCCESS)
                throw std::runtime_error("failed to create postProcessing sampler!");
        }

        blitAttachments.resize(blitAttachmentCount);
        for(uint32_t i=0;i<blitAttachments.size();i++){
            blitAttachments[i].resize(imageCount);
            for(size_t imageNumber=0; imageNumber<imageCount; imageNumber++)
            {
                createImage(        physicalDevice,
                                    device,
                                    image.Extent.width,
                                    image.Extent.height,
                                    1,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    image.Format,
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    blitAttachments[i].image[imageNumber],
                                    blitAttachments[i].imageMemory[imageNumber]);

                blitAttachments[i].imageView[imageNumber] =
                createImageView(    device,
                                    blitAttachments[i].image[imageNumber],
                                    image.Format,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    1);
            }
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
            if (vkCreateSampler(*device, &SamplerInfo, nullptr, &blitAttachments[i].sampler) != VK_SUCCESS)
                throw std::runtime_error("failed to create postProcessing sampler!");
        }

        blitAttachment.resize(imageCount);
        for(size_t imageNumber=0; imageNumber<imageCount; imageNumber++)
        {
            createImage(            physicalDevice,
                                    device,
                                    image.Extent.width,
                                    image.Extent.height,
                                    1,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    image.Format,
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT ,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    blitAttachment.image[imageNumber],
                                    blitAttachment.imageMemory[imageNumber]);

            blitAttachment.imageView[imageNumber] =
            createImageView(        device,
                                    blitAttachment.image[imageNumber],
                                    image.Format,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    1);

            transitionImageLayout(  device,
                                    graphicsQueue,
                                    commandPool,
                                    blitAttachment.image[imageNumber],
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_REMAINING_MIP_LEVELS);
        }
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
        if (vkCreateSampler(*device, &SamplerInfo, nullptr, &blitAttachment.sampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing sampler!");

        sslrAttachment.resize(imageCount);
        for(size_t imageNumber=0; imageNumber<imageCount; imageNumber++)
        {
            createImage(            physicalDevice,
                                    device,
                                    image.Extent.width,
                                    image.Extent.height,
                                    1,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    image.Format,
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT ,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    sslrAttachment.image[imageNumber],
                                    sslrAttachment.imageMemory[imageNumber]);

            sslrAttachment.imageView[imageNumber] =
            createImageView(        device,
                                    sslrAttachment.image[imageNumber],
                                    image.Format,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    1);
        }
        VkSamplerCreateInfo sslrSamplerInfo{};
            sslrSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sslrSamplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
            sslrSamplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
            sslrSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Режим адресации
            sslrSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
            sslrSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Повторение текстуры при выходе за пределы размеров изображения.
            sslrSamplerInfo.anisotropyEnable = VK_TRUE;
            sslrSamplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
            sslrSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
            sslrSamplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
            sslrSamplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
            sslrSamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
            sslrSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sslrSamplerInfo.minLod = 0.0f;
            sslrSamplerInfo.maxLod = 0.0f;
            sslrSamplerInfo.mipLodBias = 0.0f;
        if (vkCreateSampler(*device, &sslrSamplerInfo, nullptr, &sslrAttachment.sampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing sampler!");


        ssaoAttachment.resize(imageCount);
        for(size_t imageNumber=0; imageNumber<imageCount; imageNumber++)
        {
            createImage(            physicalDevice,
                                    device,
                                    image.Extent.width,
                                    image.Extent.height,
                                    1,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    image.Format,
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT ,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    ssaoAttachment.image[imageNumber],
                                    ssaoAttachment.imageMemory[imageNumber]);

            ssaoAttachment.imageView[imageNumber] =
            createImageView(        device,
                                    ssaoAttachment.image[imageNumber],
                                    image.Format,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    1);
        }
        VkSamplerCreateInfo ssaoSamplerInfo{};
            ssaoSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            ssaoSamplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
            ssaoSamplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
            ssaoSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Режим адресации
            ssaoSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
            ssaoSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Повторение текстуры при выходе за пределы размеров изображения.
            ssaoSamplerInfo.anisotropyEnable = VK_TRUE;
            ssaoSamplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
            ssaoSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
            ssaoSamplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
            ssaoSamplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
            ssaoSamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
            ssaoSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            ssaoSamplerInfo.minLod = 0.0f;
            ssaoSamplerInfo.maxLod = 0.0f;
            ssaoSamplerInfo.mipLodBias = 0.0f;
        if (vkCreateSampler(*device, &ssaoSamplerInfo, nullptr, &ssaoAttachment.sampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing sampler!");
    }

//=======================================RenderPass======================//

void postProcessing::createRenderPass()
{
    uint32_t index = 0;

    std::array<VkAttachmentDescription,2> attachments{};
        attachments[index].format = image.Format;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                       //следующие 4 параметра смотри на странице 210
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;              //в каком размещении его нужно оставить по завершению рендеринга
    index++;
        attachments[index].format = image.Format;
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

    if (vkCreateRenderPass(*device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing render pass!");
}

//===================Framebuffers===================================

void postProcessing::createFramebuffers()
{
    framebuffers.resize(image.Count);
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
            framebufferInfo.width = image.Extent.width;                                                              //ширина изображения
            framebufferInfo.height = image.Extent.height;                                                            //высота изображения
            framebufferInfo.layers = 1;                                                                                 //число слоёв
        if (vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) //создание буфера кадров
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

        std::array<VkDescriptorSetLayoutBinding,2> firstBindings{};
            firstBindings[index].binding = 0;
            firstBindings[index].descriptorCount = 1;
            firstBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            firstBindings[index].pImmutableSamplers = nullptr;
            firstBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            firstBindings[index].binding = 1;
            firstBindings[index].descriptorCount = 1;
            firstBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            firstBindings[index].pImmutableSamplers = nullptr;
            firstBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        index = 0;
        std::array<VkDescriptorSetLayoutBinding,4> secondBindings{};
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
            secondBindings[index].descriptorCount = static_cast<uint32_t>(blitAttachmentCount);
            secondBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            secondBindings[index].pImmutableSamplers = nullptr;
            secondBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        index = 0;
        std::array<VkDescriptorSetLayoutCreateInfo,2> textureLayoutInfo{};
            textureLayoutInfo[index].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo[index].bindingCount = static_cast<uint32_t>(firstBindings.size());
            textureLayoutInfo[index].pBindings = firstBindings.data();
        index++;
            textureLayoutInfo[index].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo[index].bindingCount = static_cast<uint32_t>(secondBindings.size());
            textureLayoutInfo[index].pBindings = secondBindings.data();

        if (vkCreateDescriptorSetLayout(*device, &textureLayoutInfo.at(0), nullptr, &first.DescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing descriptor set layout 1!");
        if (vkCreateDescriptorSetLayout(*device, &textureLayoutInfo.at(1), nullptr, &second.DescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing descriptor set layout 2!");
    }
    void postProcessing::createFirstGraphicsPipeline()
    {
        uint32_t index = 0;

        const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";
        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\firstPostProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\firstPostProcessingFrag.spv");
        VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
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
            viewport[index].width  = (float) image.Extent.width;
            viewport[index].height= (float) image.Extent.height;
            viewport[index].minDepth = 0.0f;
            viewport[index].maxDepth = 1.0f;
        std::array<VkRect2D,1> scissor{};
            scissor[index].offset = {0, 0};
            scissor[index].extent = image.Extent;
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
        if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &first.PipelineLayout) != VK_SUCCESS)
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
        if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &first.Pipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing graphics pipeline 1!");

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(*device, fragShaderModule, nullptr);
        vkDestroyShaderModule(*device, vertShaderModule, nullptr);
    }
    void postProcessing::createSecondGraphicsPipeline()
    {
        uint32_t index = 0;

        const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";
        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\postProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\postProcessing\\postProcessingFrag.spv");
        VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
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
            viewport[index].width  = (float) image.Extent.width;
            viewport[index].height= (float) image.Extent.height;
            viewport[index].minDepth = 0.0f;
            viewport[index].maxDepth = 1.0f;
        std::array<VkRect2D,1> scissor{};
            scissor[index].offset = {0, 0};
            scissor[index].extent = image.Extent;
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

        index=0;
        std::array<VkPushConstantRange,1> pushConstantRange{};
            pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
            pushConstantRange[index].offset = 0;
            pushConstantRange[index].size = sizeof(postProcessingPushConst);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &second.DescriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
        if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &second.PipelineLayout) != VK_SUCCESS)
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
        if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &second.Pipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessing graphics pipeline 2!");

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(*device, fragShaderModule, nullptr);
        vkDestroyShaderModule(*device, vertShaderModule, nullptr);
    }

void postProcessing::createDescriptorPool()
{
    uint32_t imageCount = image.Count;
    size_t index = 0;
    std::array<VkDescriptorPoolSize,2> firstPoolSizes;
        firstPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        firstPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        firstPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        firstPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);

    VkDescriptorPoolCreateInfo firstPoolInfo{};
        firstPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        firstPoolInfo.poolSizeCount = static_cast<uint32_t>(firstPoolSizes.size());
        firstPoolInfo.pPoolSizes = firstPoolSizes.data();
        firstPoolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(*device, &firstPoolInfo, nullptr, &first.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing descriptor pool 1!");

    index = 0;
    std::array<VkDescriptorPoolSize,4> secondPoolSizes;
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
        secondPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(blitAttachmentCount*imageCount);
    VkDescriptorPoolCreateInfo secondPoolInfo{};
        secondPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        secondPoolInfo.poolSizeCount = static_cast<uint32_t>(secondPoolSizes.size());
        secondPoolInfo.pPoolSizes = secondPoolSizes.data();
        secondPoolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(*device, &secondPoolInfo, nullptr, &second.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessing descriptor pool 2!");
}

void postProcessing::createDescriptorSets(DeferredAttachments Attachments)
{
    uint32_t imageCount = image.Count;
    first.DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> firstLayouts(imageCount, first.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo firstAllocInfo{};
        firstAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        firstAllocInfo.descriptorPool = first.DescriptorPool;
        firstAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        firstAllocInfo.pSetLayouts = firstLayouts.data();
    if (vkAllocateDescriptorSets(*device, &firstAllocInfo, first.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate postProcessing descriptor sets 1!");

    for (size_t image = 0; image < imageCount; image++)
    {
        uint32_t index = 0;
        std::array<VkDescriptorImageInfo,2> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments.blur->imageView[image];
            imageInfo[index].sampler = Attachments.blur->sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = ssaoAttachment.imageView[image];
            imageInfo[index].sampler = ssaoAttachment.sampler;

        index = 0;
        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = first.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = first.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    second.DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, second.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = second.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(*device, &allocInfo, second.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate postProcessing descriptor sets 2!");

    for (size_t image = 0; image < imageCount; image++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorImageInfo, 4> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments.image->imageView[image];
            imageInfo[index].sampler = Attachments.image->sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = this->Attachments[0].imageView[image];
            imageInfo[index].sampler = this->Attachments[0].sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = sslrAttachment.imageView[image];
            imageInfo[index].sampler = sslrAttachment.sampler;

        index = 0;
        std::array<VkDescriptorImageInfo, blitAttachmentCount> blitImageInfo;
            for(uint32_t i=0;i<blitImageInfo.size();i++,index++){
                blitImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                blitImageInfo[index].imageView = blitAttachments[i].imageView[image];
                blitImageInfo[index].sampler = blitAttachments[i].sampler;
            }


        index = 0;
        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
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
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(blitImageInfo.size());
            descriptorWrites[index].pImageInfo = blitImageInfo.data();
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void postProcessing::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    std::array<VkClearValue, 2> ClearValues{};
        ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        ClearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffers, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, first.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, first.PipelineLayout, 0, 1, &first.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers, 6, 1, 0, 0);

    vkCmdNextSubpass(commandBuffers, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdNextSubpass(commandBuffers, VK_SUBPASS_CONTENTS_INLINE);

        postProcessingPushConst pushConst{};
            pushConst.blitFactor = blitFactor;
        vkCmdPushConstants(commandBuffers, second.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(postProcessingPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, second.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, second.PipelineLayout, 0, 1, &second.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers);
}

std::vector<attachments>        &postProcessing::getBlitAttachments(){return blitAttachments;}
attachments                     &postProcessing::getBlitAttachment(){return blitAttachment;}
attachments                     &postProcessing::getSSLRAttachment(){return sslrAttachment;}
attachments                     &postProcessing::getSSAOAttachment(){return ssaoAttachment;}

VkSwapchainKHR                  &postProcessing::SwapChain(){return swapChain;}
uint32_t                        &postProcessing::SwapChainImageCount(){return image.Count;}
VkFormat                        &postProcessing::SwapChainImageFormat(){return image.Format;}
VkExtent2D                      &postProcessing::SwapChainImageExtent(){return image.Extent;}
