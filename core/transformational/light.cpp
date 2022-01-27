#include "light.h"
#include <fstream>
#include "object.h"
#include "core/operations.h"
#include "gltfmodel.h"
#include "camera.h"

light<spotLight>::light(VkApplication *app, uint32_t type)
    : app(app), type(type)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
}

light<spotLight>::~light()
{
    deleteLight();
}

void light<spotLight>::cleanup()
{
    for (size_t i = 0; i < lightUniformBuffers.size(); i++)
    {
        vkDestroyBuffer(app->getDevice(), lightUniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), lightUniformBuffersMemory[i], nullptr);
    }

    if(enableShadow)
    {
        depthAttachment.deleteAttachment(&app->getDevice());
        for(uint32_t i=0;i<shadowMapFramebuffer.size();i++)
        {
            vkDestroyFramebuffer(app->getDevice(), shadowMapFramebuffer.at(i),nullptr);
        }
        for(uint32_t i=0;i<shadowCommandPool.size();i++)
        {
            vkFreeCommandBuffers(app->getDevice(), shadowCommandPool.at(i), static_cast<uint32_t>(shadowCommandBuffer.at(i).size()), shadowCommandBuffer.at(i).data());
        }

        vkDestroyPipeline(app->getDevice(), shadowPipeline, nullptr);
        vkDestroyPipelineLayout(app->getDevice(),shadowPipelineLayout,nullptr);
        vkDestroyRenderPass(app->getDevice(), shadowRenerPass, nullptr);

        vkDestroyDescriptorSetLayout(app->getDevice(), descriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(app->getDevice(), uniformBlockSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(app->getDevice(), uniformBufferSetLayout, nullptr);
        vkDestroyDescriptorPool(app->getDevice(), descriptorPool, nullptr);

        vkDestroySampler(app->getDevice(), shadowSampler, nullptr);
        for(size_t i = 0; i < shadowCommandPool.size(); i++)
        {
            vkDestroyCommandPool(app->getDevice(), shadowCommandPool.at(i), nullptr);
        }
    }
}

void light<spotLight>::deleteLight()
{
    if(deleted == false)
    {
        cleanup();
        deleted = true;
    }
}

void light<spotLight>::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void light<spotLight>::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void light<spotLight>::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void light<spotLight>::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void light<spotLight>::updateViewMatrix()
{
    glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),-m_translate);
    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
    {
        rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    }
    glm::mat4x4 scaleMatrix = glm::scale(glm::mat4x4(1.0f),m_scale);
    modelMatrix = m_globalTransform * translateMatrix * rotateMatrix * scaleMatrix;
    viewMatrix = glm::inverse(modelMatrix);
}

void light<spotLight>::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void light<spotLight>::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void light<spotLight>::createShadowImage()
{
    mipLevels = 1;
    VkFormat shadowFormat = findDepthFormat(app);
    createImage(app,SHADOW_MAP_WIDTH,SHADOW_MAP_HEIGHT,mipLevels,VK_SAMPLE_COUNT_1_BIT,shadowFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthAttachment.image, depthAttachment.imageMemory);
}

void light<spotLight>::createShadowImageView()
{
    VkFormat shadowFormat = findDepthFormat(app);
    depthAttachment.imageView = createImageView(app,depthAttachment.image, shadowFormat, VK_IMAGE_ASPECT_DEPTH_BIT, mipLevels);
}

void light<spotLight>::createLightPVM(const glm::mat4x4 & projection)
{
    projectionMatrix = projection;
    viewMatrix = glm::mat4x4(1.0f);
    modelMatrix = glm::mat4x4(1.0f);
}

void light<spotLight>::createShadowSampler()
{
    /* Сэмплеры настраиваются через VkSamplerCreateInfoструктуру, которая определяет
     * все фильтры и преобразования, которые она должна применять.*/

    float mipLevel = 1.0f;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
    samplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Режим адресации
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Повторение текстуры при выходе за пределы размеров изображения.
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
    samplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
    samplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.minLod = static_cast<float>(mipLevel*mipLevels);
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    samplerInfo.mipLodBias = 0.0f; // Optional

    if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &shadowSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void light<spotLight>::createShadowRenderPass()
{
    VkAttachmentDescription attachments[1];
       // Depth attachment (shadow map)
       attachments[0].format =  findDepthFormat(app);
       attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
       attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
       attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
       attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
       attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
       attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
       attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
       attachments[0].flags = 0;

       // Attachment references from subpasses
       VkAttachmentReference depth_ref;
       depth_ref.attachment = 0;
       depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

       // Subpass 0: shadow map rendering
       VkSubpassDescription subpass[1];
       subpass[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
       subpass[0].flags = 0;
       subpass[0].inputAttachmentCount = 0;
       subpass[0].pInputAttachments = NULL;
       subpass[0].colorAttachmentCount = 0;
       subpass[0].pColorAttachments = NULL;
       subpass[0].pResolveAttachments = NULL;
       subpass[0].pDepthStencilAttachment = &depth_ref;
       subpass[0].preserveAttachmentCount = 0;
       subpass[0].pPreserveAttachments = NULL;

    // Create render pass
    VkRenderPassCreateInfo renderPassInfo;
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.pNext = NULL;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = subpass;
    renderPassInfo.dependencyCount = 0;
    renderPassInfo.pDependencies = NULL;
    renderPassInfo.flags = 0;

    vkCreateRenderPass(app->getDevice(), &renderPassInfo, NULL, &shadowRenerPass);
}

void light<spotLight>::createShadowMapFramebuffer()
{
    shadowMapFramebuffer.resize(imageCount);
    for (size_t i = 0; i < shadowMapFramebuffer.size(); i++)
    {
        VkFramebufferCreateInfo framebufferInfo;
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.pNext = NULL;
        framebufferInfo.renderPass = shadowRenerPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &depthAttachment.imageView;
        framebufferInfo.width = SHADOW_MAP_WIDTH;
        framebufferInfo.height = SHADOW_MAP_HEIGHT;
        framebufferInfo.layers = 1;
        framebufferInfo.flags = 0;

        vkCreateFramebuffer(app->getDevice(), &framebufferInfo, NULL, &shadowMapFramebuffer.at(i));
    }
}

void light<spotLight>::createShadowCommandPool()
{
    shadowCommandPool.resize(LIGHT_COMMAND_POOLS);
    shadowCommandBuffer.resize(LIGHT_COMMAND_POOLS);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = app->getQueueFamilyIndices().graphicsFamily.value();      //задаёт семейство очередей, в которые будет передаваться созданные командные буферы
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;           //задаёт флаги, определяющие поведение пула и командных буферов, выделяемых из него

    for(size_t i = 0; i < shadowCommandPool.size(); i++)
    {
        if (vkCreateCommandPool(app->getDevice(), &poolInfo, nullptr, &shadowCommandPool.at(i)) != VK_SUCCESS) //создание пула команд
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }
}

void light<spotLight>::createShadowDescriptorSetLayout()
{
    /* Нам нужно предоставить подробную информацию о каждой привязке дескриптора,
     * используемой в шейдерах для создания конвейера, точно так же, как мы должны
     * были сделать для каждого атрибута вершины и ее locationиндекса. Мы создадим
     * новую функцию для определения всей этой информации с именем createDescriptorSetLayout*/

    VkDescriptorSetLayoutBinding lightUboLayoutBinding={};
    lightUboLayoutBinding.binding = 1;
    lightUboLayoutBinding.descriptorCount = 1;
    lightUboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightUboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    lightUboLayoutBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {lightUboLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkDescriptorSetLayoutBinding uniformBufferLayoutBinding{};
    uniformBufferLayoutBinding.binding = 0;
    uniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferLayoutBinding.descriptorCount = 1;
    uniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uniformBufferLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
    uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    uniformBufferLayoutInfo.bindingCount = 1;
    uniformBufferLayoutInfo.pBindings = &uniformBufferLayoutBinding;

    if (vkCreateDescriptorSetLayout(app->getDevice(), &uniformBufferLayoutInfo, nullptr, &uniformBufferSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkDescriptorSetLayoutBinding uniformBlockLayoutBinding{};
    uniformBlockLayoutBinding.binding = 0;
    uniformBlockLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBlockLayoutBinding.descriptorCount = 1;
    uniformBlockLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uniformBlockLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo uniformBlockLayoutInfo{};
    uniformBlockLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    uniformBlockLayoutInfo.bindingCount = 1;
    uniformBlockLayoutInfo.pBindings = &uniformBlockLayoutBinding;

    if (vkCreateDescriptorSetLayout(app->getDevice(), &uniformBlockLayoutInfo, nullptr, &uniformBlockSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void light<spotLight>::createShadowDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes(1);
    size_t index = 0;

    for(size_t i=0;i<1;i++)
    {
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
        index++;
    }

    //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
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

void light<spotLight>::createShadowDescriptorSets()
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

    //Наборы дескрипторов уже выделены, но дескрипторы внутри еще нуждаются в настройке.
    //Теперь мы добавим цикл для заполнения каждого дескриптора:
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorBufferInfo lightBufferInfo;
        lightBufferInfo.buffer = lightUniformBuffers.at(i);
        lightBufferInfo.offset = 0;
        lightBufferInfo.range = sizeof(LightUniformBufferObject);
        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 1;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &lightBufferInfo;

        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void light<spotLight>::createShadowPipeline()
{
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\shadow\\shad.spv");
    VkShaderModule vertShaderModule = createShaderModule(app,vertShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo};

    auto bindingDescription = gltfModel::Vertex::getBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();                            //указатель на массив соответствующийх структуру

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                        //без него кажда полоса и веер потребуют отдельной команды вывода.

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) SHADOW_MAP_WIDTH;
    viewport.height = (float) SHADOW_MAP_HEIGHT;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent.width = SHADOW_MAP_WIDTH;
    scissor.extent.height = SHADOW_MAP_HEIGHT;
    scissor.offset.x = 0;
    scissor.offset.y = 0;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;                             //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 4.0f;
    rasterizer.depthBiasSlopeFactor = 1.5f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                         //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                       //Optional
    colorBlending.attachmentCount = 1;                                              //количество подключений
    colorBlending.pAttachments = &colorBlendAttachment;                             //массив подключений

    std::array<VkDescriptorSetLayout,3> SetLayouts = {descriptorSetLayout,uniformBufferSetLayout,uniformBufferSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &shadowPipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 1;                                            //число структур в массиве структур
    pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
    pipelineInfo.pVertexInputState = &vertexInputInfo;                      //вершинный ввод
    pipelineInfo.pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
    pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
    pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
    pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
    pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
    pipelineInfo.layout = shadowPipelineLayout;                             //
    pipelineInfo.renderPass = shadowRenerPass;                              //проход рендеринга
    pipelineInfo.subpass = 0;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void light<spotLight>::createUniformBuffers()
{
    VkDeviceSize bufferSize = sizeof(LightUniformBufferObject);

    lightUniformBuffers.resize(imageCount);
    lightUniformBuffersMemory.resize(imageCount);

    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app,bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, lightUniformBuffers[i], lightUniformBuffersMemory[i]);
    }
}

void light<spotLight>::updateUniformBuffer(uint32_t currentImage)
{
    LightUniformBufferObject ubo;
    ubo.position = modelMatrix * glm::vec4(0.0f,0.0f,0.0f,1.0f);
    ubo.projView = projectionMatrix*viewMatrix;
    ubo.lightColor = lightColor;
    ubo.type = type;
    ubo.enableShadow = static_cast<uint32_t>(enableShadow);

    void* data;
    vkMapMemory(app->getDevice(), lightUniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(LightUniformBufferObject));
    vkUnmapMemory(app->getDevice(), lightUniformBuffersMemory[currentImage]);
}

void light<spotLight>::createShadow(uint32_t commandPoolsCount)
{
    enableShadow = true;

    setCommandPoolsCount(commandPoolsCount);
    createShadowCommandPool();
    createShadowSampler();
    createShadowRenderPass();
    createShadowImage();
    createShadowImageView();
    createShadowMapFramebuffer();
    createShadowDescriptorSetLayout();
    createShadowPipeline();
    createShadowDescriptorPool();
    createShadowDescriptorSets();
    for(size_t j=0;j<shadowCommandPool.size();j++)
    {
        createShadowCommandBuffers(j);
    }
}

void light<spotLight>::createShadowCommandBuffers(uint32_t number)
{
    shadowCommandBuffer[number].resize(imageCount);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = shadowCommandPool[number];
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount =  imageCount;

    if (vkAllocateCommandBuffers(app->getDevice(), &allocInfo, shadowCommandBuffer[number].data()) != VK_SUCCESS)
    {throw std::runtime_error("failed to allocate command buffers!");}
}

void light<spotLight>::updateShadowCommandBuffers(uint32_t number, uint32_t i, std::vector<object *> & object3D)
{
    VkClearValue clearValues{};
    clearValues.depthStencil.depth = 1.0f;
    clearValues.depthStencil.stencil = 0;

        /* Прежде чем вы сможете начать записывать команды в командный буфер, вам
         * нужно начать командный буфер, т.е. просто сбростить к начальному состоянию.
         * Для этого выозвите функцию vkBeginCommandBuffer*/

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                                            //поле для передачи информации о том, как будет использоваться этот командный буфер (смотри страницу 102)
        beginInfo.pInheritanceInfo = nullptr;                           //используется при начале вторичного буфера, для того чтобы определить, какие состояния наследуются от первичного командного буфера, который его вызовет
        if (vkBeginCommandBuffer(shadowCommandBuffer[number][i], &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.pNext = NULL;
            renderPassInfo.renderPass = shadowRenerPass;
            renderPassInfo.framebuffer = shadowMapFramebuffer[i];
            renderPassInfo.renderArea.offset.x = 0;
            renderPassInfo.renderArea.offset.y = 0;
            renderPassInfo.renderArea.extent.width = SHADOW_MAP_WIDTH;
            renderPassInfo.renderArea.extent.height = SHADOW_MAP_HEIGHT;
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearValues;

            vkCmdBeginRenderPass(shadowCommandBuffer[number][i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

                    VkViewport viewport;
                    viewport.height = SHADOW_MAP_HEIGHT;
                    viewport.width = SHADOW_MAP_WIDTH;
                    viewport.minDepth = 0.0f;
                    viewport.maxDepth = 1.0f;
                    viewport.x = 0;
                    viewport.y = 0;
                    vkCmdSetViewport(shadowCommandBuffer[number][i], 0, 1, &viewport);

                    VkRect2D scissor;
                    scissor.extent.width = SHADOW_MAP_WIDTH;
                    scissor.extent.height = SHADOW_MAP_HEIGHT;
                    scissor.offset.x = 0;
                    scissor.offset.y = 0;
                    vkCmdSetScissor(shadowCommandBuffer[number][i], 0, 1, &scissor);

                    for(size_t j = 0; j<object3D.size() ;j++)
                    {
                        VkDeviceSize offsets[1] = { 0 };

                        vkCmdBindPipeline(shadowCommandBuffer[number][i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);

                        vkCmdBindVertexBuffers(shadowCommandBuffer[number][i], 0, 1, & object3D[j]->getModel()->vertices.buffer, offsets);
                        if (object3D[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
                        {
                            vkCmdBindIndexBuffer(shadowCommandBuffer[number][i],  object3D[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                        }

                        for (auto node : object3D[j]->getModel()->nodes)
                        {
//                            glm::vec3 position = glm::vec3(object3D[j]->getTransformation()*node->matrix*glm::vec4(0.0f,0.0f,0.0f,1.0f));
//                            if(glm::length(position-camera->getTranslate())<object3D[j]->getVisibilityDistance()){
//                                renderNode(node,shadowCommandBuffer[number][i],descriptorSets[i],object3D[j]->getDescriptorSet()[i]);
//                            }

                            renderNode(node,shadowCommandBuffer[number][i],descriptorSets[i],object3D[j]->getDescriptorSet()[i]);
                        }
                    }

            vkCmdEndRenderPass(shadowCommandBuffer[number][i]);
        if (vkEndCommandBuffer(shadowCommandBuffer[number][i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
}

void light<spotLight>::renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
            const std::vector<VkDescriptorSet> descriptorsets =
            {
                descriptorSet,
                objectDescriptorSet,
                node->mesh->uniformBuffer.descriptorSet
            };
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 0, static_cast<uint32_t>(descriptorsets.size()), descriptorsets.data(), 0, NULL);

            if (primitive->hasIndices)
            {
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            }
            else
            {
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
            }
        }
    }

    for (auto child : node->children)
    {
        renderNode(child, commandBuffer, descriptorSet,objectDescriptorSet);
    }
}


void                            light<spotLight>::setCommandPoolsCount(const int & count){this->LIGHT_COMMAND_POOLS = count;}
void                            light<spotLight>::setLightColor(const glm::vec4 &color){this->lightColor = color;}
void                            light<spotLight>::setImageCount(uint32_t imageCount){this->imageCount=imageCount;}
void                            light<spotLight>::setCamera(class camera *camera){this->camera=camera;}

glm::mat4x4                     light<spotLight>::getViewMatrix() const {return viewMatrix;}
glm::mat4x4                     light<spotLight>::getModelMatrix() const {return modelMatrix;}
glm::vec3                       light<spotLight>::getTranslate() const {return m_translate;}

uint32_t                        light<spotLight>::getWidth() const {return SHADOW_MAP_WIDTH;}
uint32_t                        light<spotLight>::getHeight() const {return SHADOW_MAP_HEIGHT;}
glm::vec4                       light<spotLight>::getLightColor() const {return lightColor;}
bool                            light<spotLight>::getShadowEnable() const{return enableShadow;}

VkImage                         & light<spotLight>::getShadowImage(){return depthAttachment.image;}
VkDeviceMemory                  & light<spotLight>::getShadowImageMemory(){return depthAttachment.imageMemory;}
VkImageView                     & light<spotLight>::getImageView(){return depthAttachment.imageView;}
VkSampler                       & light<spotLight>::getSampler(){return shadowSampler;}
VkRenderPass                    & light<spotLight>::getShadowRenerPass(){return shadowRenerPass;}
std::vector<VkFramebuffer>      & light<spotLight>::getShadowMapFramebuffer(){return shadowMapFramebuffer;}
VkPipeline                      & light<spotLight>::getShadowPipeline(){return shadowPipeline;}
VkCommandPool                   & light<spotLight>::getShadowCommandPool(uint32_t number){return shadowCommandPool[number];}
VkPipelineLayout                & light<spotLight>::getShadowPipelineLayout(){return shadowPipelineLayout;}
std::vector<VkCommandBuffer>    & light<spotLight>::getCommandBuffer(uint32_t number){return shadowCommandBuffer[number];}
std::vector<VkBuffer>           & light<spotLight>::getLightUniformBuffers(){return lightUniformBuffers;}
std::vector<VkDeviceMemory>     & light<spotLight>::getLightUniformBuffersMemory(){return lightUniformBuffersMemory;}

//======================================================================================================================//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//======================================================================================================================//

light<pointLight>::light(VkApplication *app, std::vector<light<spotLight> *> & lightSource) : lightSource(lightSource)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    number = lightSource.size();

    glm::mat4x4 Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
    Proj[1][1] *= -1;

    int index = number;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,1.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.3f,0.6f,0.9f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.6f,0.9f,0.3f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(180.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.9f,0.3f,0.6f,1.0f));
}

light<pointLight>::~light(){}

glm::vec4       light<pointLight>::getLightColor() const {return lightColor;}
uint32_t        light<pointLight>::getNumber() const {return number;}
glm::vec3       light<pointLight>::getTranslate() const {return m_translate;}

void light<pointLight>::setLightColor(const glm::vec4 &color)
{
    this->lightColor = color;
    for(uint32_t i=number;i<number+6;i++)
    {
        lightSource.at(i)->setLightColor(color);
    }
}

void light<pointLight>::setCamera(class camera *camera)
{
    for(uint32_t i=number;i<number+6;i++)
    {
        lightSource.at(i)->setCamera(camera);
    }
}

void light<pointLight>::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void light<pointLight>::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void light<pointLight>::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void light<pointLight>::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void light<pointLight>::updateViewMatrix()
{
    glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),-m_translate);
    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
    {
        rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    }
    glm::mat4x4 scaleMatrix = glm::scale(glm::mat4x4(1.0f),m_scale);

    glm::mat4x4 localMatrix = m_globalTransform * translateMatrix * rotateMatrix * scaleMatrix;

    for(uint32_t i=number;i<number+6;i++)
    {
        lightSource.at(i)->setGlobalTransform(localMatrix);
    }
}

void light<pointLight>::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void light<pointLight>::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    updateViewMatrix();
}

