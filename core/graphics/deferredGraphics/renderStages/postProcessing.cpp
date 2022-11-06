#include "postProcessing.h"
#include "../bufferObjects.h"

#include <iostream>
#include <cstdint>          // нужна для UINT32_MAX
#include <array>
#include <algorithm>        // нужна для std::min/std::max

postProcessingGraphics::postProcessingGraphics()
{}

void postProcessingGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, uint32_t graphicsFamily, uint32_t presentFamily)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
    this->queueFamilyIndices = {graphicsFamily,presentFamily};
}
void postProcessingGraphics::setImageProp(imageInfo* pInfo)
{
    this->image = *pInfo;
}

void postProcessingGraphics::setSwapChain(VkSwapchainKHR* swapChain)
{
    this->swapChain = swapChain;
}

void postProcessingGraphics::setBlurAttachment(attachments *blurAttachment)
{
    this->blurAttachment = blurAttachment;
}
void postProcessingGraphics::setBlitAttachments(uint32_t blitAttachmentCount, attachments* blitAttachments, float blitFactor)
{
    postProcessing.blitFactor = blitFactor;
    postProcessing.blitAttachmentCount = blitAttachmentCount;
    this->blitAttachments = blitAttachments;
}
void postProcessingGraphics::setSSLRAttachment(attachments* sslrAttachment)
{
    this->sslrAttachment = sslrAttachment;
}
void postProcessingGraphics::setSSAOAttachment(attachments* ssaoAttachment)
{
    this->ssaoAttachment = ssaoAttachment;
}

void postProcessingGraphics::setTransparentLayersCount(uint32_t TransparentLayersCount)
{
    postProcessing.transparentLayersCount = TransparentLayersCount;
}

void postProcessingGraphics::PostProcessing::Destroy(VkDevice* device)
{
    vkDestroyPipeline(*device, Pipeline, nullptr);
    vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
}

void postProcessingGraphics::destroy()
{
    postProcessing.Destroy(device);

    vkDestroyRenderPass(*device, renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        vkDestroyFramebuffer(*device, framebuffers[i],nullptr);

    uint32_t imageCount = image.Count;
    for(size_t i=0; i<swapChainAttachments.size(); i++)
        for(size_t image=0; image <imageCount;image++)
            vkDestroyImageView(*device,swapChainAttachments[i].imageView[image],nullptr);
    vkDestroySwapchainKHR(*device, *swapChain, nullptr);
}

void postProcessingGraphics::setExternalPath(const std::string &path)
{
    postProcessing.ExternalPath = path;
}

void postProcessingGraphics::createAttachments(GLFWwindow* window, SwapChainSupportDetails swapChainSupport, VkSurfaceKHR* surface)
{
    this->surface = surface;

    createSwapChain(window, swapChainSupport);
    createImageViews();
}
    //Создание цепочки обмена
    void postProcessingGraphics::createSwapChain(GLFWwindow* window, SwapChainSupportDetails swapChainSupport)
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

        QueueFamilyIndices indices = queueFamilyIndices;
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

        if (vkCreateSwapchainKHR(*device, &createInfo, nullptr, swapChain) != VK_SUCCESS)     //функция дял создания цепочки обмена, устройство с которым связан список показа передаётся в параметре device
            throw std::runtime_error("failed to create swap chain!");                         //информация о списке показа передаётся в виде структуры VkSwapchainCreateInfoKHR которая определена выше

        vkGetSwapchainImagesKHR(*device, *swapChain, &imageCount, nullptr);                   //записываем дескриптор изображений представляющий элементы в списке показа

        swapChainAttachments.resize(swapChainAttachmentCount);
        for(size_t i=0;i<swapChainAttachments.size();i++)
        {
            swapChainAttachments[i].image.resize(imageCount);
            swapChainAttachments[i].imageView.resize(imageCount);
            swapChainAttachments[i].setSize(imageCount);
            vkGetSwapchainImagesKHR(*device, *swapChain, &imageCount, swapChainAttachments[i].image.data());    //получаем дескрипторы, на них мы будем ссылаться при рендеринге
        }                   
    }

    void postProcessingGraphics::createImageViews()
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

void postProcessingGraphics::createRenderPass()
{
    uint32_t index = 0;
    std::array<VkAttachmentDescription,1> attachments{};
        attachments[index].format = image.Format;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                       //следующие 4 параметра смотри на странице 210
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;              //в каком размещении его нужно оставить по завершению рендеринга

    index = 0;
    std::array<VkAttachmentReference,1> attachmentRef;
        attachmentRef[index].attachment = 0;
        attachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    index = 0;
    std::array<VkSubpassDescription,1> subpass{};
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(attachmentRef.size());
        subpass[index].pColorAttachments = attachmentRef.data();

    index = 0;
    std::array<VkSubpassDependency,1> dependency{};                                                                                        //зависимости
        dependency[index].srcSubpass = VK_SUBPASS_EXTERNAL;                                                                                //ссылка из исходного прохода (создавшего данные)
        dependency[index].dstSubpass = 0;                                                                                                  //в целевой подпроход (поглощающий данные)
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;                                                                 //задаёт как стадии конвейера в исходном проходе создают данные
        dependency[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependency.size());
        renderPassInfo.pDependencies = dependency.data();
    if (vkCreateRenderPass(*device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessingGraphics render pass!");
}

void postProcessingGraphics::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for (size_t i = 0; i < framebuffers.size(); i++)
    {
        uint32_t index = 0;
        std::vector<VkImageView> attachments(swapChainAttachments.size());
            for(size_t j=0;j<swapChainAttachments.size();j++){
                attachments.at(index) = swapChainAttachments.at(j).imageView[i]; index++;}

        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        if (vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessingGraphics framebuffer!");
    }
}

void postProcessingGraphics::createPipelines()
{
    postProcessing.createDescriptorSetLayout(device);
    postProcessing.createPipeline(device,&image,&renderPass);
}
    void postProcessingGraphics::PostProcessing::createDescriptorSetLayout(VkDevice* device)
    {
        uint32_t index = 0;
        std::array<VkDescriptorSetLayoutBinding,7> bindings{};
            bindings[index].binding = index;
            bindings[index].descriptorCount = 1;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].pImmutableSamplers = nullptr;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            bindings[index].binding = index;
            bindings[index].descriptorCount = 1;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].pImmutableSamplers = nullptr;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            bindings[index].binding = index;
            bindings[index].descriptorCount = 1;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].pImmutableSamplers = nullptr;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            bindings[index].binding = index;
            bindings[index].descriptorCount = static_cast<uint32_t>(blitAttachmentCount);
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].pImmutableSamplers = nullptr;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            bindings[index].binding = index;
            bindings[index].descriptorCount = static_cast<uint32_t>(3);
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].pImmutableSamplers = nullptr;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            bindings[index].binding = index;
            bindings[index].descriptorCount = static_cast<uint32_t>(1);
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].pImmutableSamplers = nullptr;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        index++;
            bindings[index].binding = index;
            bindings[index].descriptorCount = transparentLayersCount;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].pImmutableSamplers = nullptr;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
            textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            textureLayoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(*device, &textureLayoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessingGraphics descriptor set layout!");
    }
    void postProcessingGraphics::PostProcessing::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
    {
        uint32_t index = 0;

        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\postProcessing\\postProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\postProcessing\\postProcessingFrag.spv");
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
            viewport[index].width  = (float) pInfo->Extent.width;
            viewport[index].height= (float) pInfo->Extent.height;
            viewport[index].minDepth = 0.0f;
            viewport[index].maxDepth = 1.0f;
        std::array<VkRect2D,1> scissor{};
            scissor[index].offset = {0, 0};
            scissor[index].extent = pInfo->Extent;
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
            pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
        if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessingGraphics pipeline layout 2!");

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
            pipelineInfo[index].layout = PipelineLayout;
            pipelineInfo[index].renderPass = *pRenderPass;
            pipelineInfo[index].subpass = 0;
            pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo[index].pDepthStencilState = &depthStencil;
        if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create postProcessingGraphics graphics pipeline 2!");

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(*device, fragShaderModule, nullptr);
        vkDestroyShaderModule(*device, vertShaderModule, nullptr);
    }

void postProcessingGraphics::createDescriptorPool()
{
    uint32_t imageCount = image.Count;
    uint32_t index = 0;
    std::array<VkDescriptorPoolSize,7> poolSizes;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(postProcessing.blitAttachmentCount*imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(3*imageCount);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(3);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(postProcessing.transparentLayersCount*imageCount);
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &postProcessing.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create postProcessingGraphics descriptor pool 2!");
}

void postProcessingGraphics::createDescriptorSets()
{
    postProcessing.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, postProcessing.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = postProcessing.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(*device, &allocInfo, postProcessing.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate postProcessingGraphics descriptor sets 2!");

}

void postProcessingGraphics::updateDescriptorSets(DeferredAttachments Attachments, std::vector<DeferredAttachments> transparentLayers)
{
    for (size_t image = 0; image < this->image.Count; image++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorImageInfo, 3> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = Attachments.image.imageView[image];
            imageInfo[index].sampler = Attachments.image.sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = blurAttachment->imageView[image];
            imageInfo[index].sampler = blurAttachment->sampler;
        index++;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = sslrAttachment->imageView[image];
            imageInfo[index].sampler = sslrAttachment->sampler;

        index = 0;
        std::vector<VkDescriptorImageInfo> blitImageInfo(postProcessing.blitAttachmentCount);
            for(uint32_t i=0;i<blitImageInfo.size();i++,index++){
                blitImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                blitImageInfo[index].imageView = blitAttachments[i].imageView[image];
                blitImageInfo[index].sampler = blitAttachments[i].sampler;
            }

        index = 0;
        std::vector<VkDescriptorImageInfo> transparentLayersInfo(transparentLayers.size());
            for(uint32_t i=0;i<transparentLayersInfo.size();i++,index++){
                transparentLayersInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                transparentLayersInfo[index].imageView = transparentLayers[i].image.imageView[image];
                transparentLayersInfo[index].sampler = transparentLayers[i].image.sampler;
            }

        index = 0;
        std::vector<VkDescriptorImageInfo> transparentLayersDepthInfo(postProcessing.transparentLayersCount);
            for(uint32_t i=0;i<transparentLayersDepthInfo.size();i++,index++){
                transparentLayersDepthInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                transparentLayersDepthInfo[index].imageView = transparentLayers[i].depth.imageView;
                transparentLayersDepthInfo[index].sampler = transparentLayers[i].depth.sampler;
            }

        VkDescriptorImageInfo imageDepthInfo;
            imageDepthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageDepthInfo.imageView = Attachments.depth.imageView;
            imageDepthInfo.sampler = Attachments.depth.sampler;

        index = 0;
        std::array<VkWriteDescriptorSet, 7> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &imageInfo[index];
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(blitImageInfo.size());
            descriptorWrites[index].pImageInfo = blitImageInfo.data();
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(transparentLayersInfo.size());
            descriptorWrites[index].pImageInfo = transparentLayersInfo.data();
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(1);
            descriptorWrites[index].pImageInfo = &imageDepthInfo;
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(transparentLayersDepthInfo.size());
            descriptorWrites[index].pImageInfo = transparentLayersDepthInfo.data();
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void postProcessingGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    std::array<VkClearValue, 1> ClearValues{};
        ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffers, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        postProcessingPushConst pushConst{};
            pushConst.blitFactor = postProcessing.blitFactor;
        vkCmdPushConstants(commandBuffers, postProcessing.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(postProcessingPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.PipelineLayout, 0, 1, &postProcessing.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers);
}

uint32_t                        &postProcessingGraphics::SwapChainImageCount(){return image.Count;}
VkFormat                        &postProcessingGraphics::SwapChainImageFormat(){return image.Format;}
