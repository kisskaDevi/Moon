#include "operations.h"
#include <set>
#include <fstream>

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(VkApplication* app, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    /* Буферы - это простейший тип ресурсов, но они могут использоваться в Vulkan для большого числа различных целей
     * Они используются для хранеия линейных структурированных и неструктурированных данных, которые могут иметь формат
     * или просто быть байтами в памяти.*/

    VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;                                                                                                 //задаёт размер буфера в байтах
        bufferInfo.usage = usage;                                                                                               //поле говорит Vulkan, как мысобираемся использовать буфер, и является набором битов из перечисления VkBufferUsageFlagBits (страница 53)
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;                                                                     //сообщает о том, как буфер будет использован в различных очередях команд. VK_SHARING_MODE_EXCLUSIVE говорит о том что данный буфер будет испольован только на одной очереди

    if (vkCreateBuffer(app->getDevice(), &bufferInfo, nullptr, &buffer) != VK_SUCCESS)                                          //функция содания буфера
        throw std::runtime_error("failed to create buffer!");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(app->getDevice(), buffer, &memRequirements);                                                  //возвращает требования к памяти для указанного объекта Vulkan

    VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;                                                                        //размеры выделяемой памяти
        allocInfo.memoryTypeIndex = findMemoryType(app->getPhysicalDevice(), memRequirements.memoryTypeBits, properties);       //типа памяти, который является индексом в массив типов памяти, вовращаемых vkGetPhisicalDeviceMemoryProperties()
    if (vkAllocateMemory(app->getDevice(), &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)                                   //выделение памяти устройства
        throw std::runtime_error("failed to allocate buffer memory!");

    vkBindBufferMemory(app->getDevice(), buffer, bufferMemory, 0);
}

VkCommandBuffer beginSingleTimeCommands(VkApplication* app)
{
    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;                                                                      //уровень командных буферов, которые вы хотите выделит. Vulkan позволяет первичным (primary) вызывать вторичные (secondary) командные буферы
        allocInfo.commandPool = app->getCommandPool().at(0);                                                                    //дескриптор созданного ранее командного пула
        allocInfo.commandBufferCount = 1;                                                                                       //число командных буферов, которые мы хотим выделить
    vkAllocateCommandBuffers(app->getDevice(), &allocInfo, &commandBuffer);                                                     //выделение командного буфера
                                                                                                                                //Прежде чем вы сможете начать записывать команды в командный буфер, вам нужно начать командный буфер, т.е. просто сброситть к начальному состоянию
    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;                                                          //информация о том как будет использоваться этот буфер команд (все флаги смотри на 102 станице)
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;                                                          //VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT - значит что командный буфер будет записан, один раз выполнен и затем уничтожен или преиспользован
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkApplication* app, VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);                                                                                          //после выполнения Vulkan завершает всю работу, которая необходима, чтобы буфер стал готовым к выполнению

    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(app->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);                                                     //эта команда может передать один или несколько командных буферов на выполенение устройством
    vkQueueWaitIdle(app->getGraphicsQueue());                                                                                   //ждём пока очередь передачи не станет свободной

    vkFreeCommandBuffers(app->getDevice(), app->getCommandPool().at(0), 1, &commandBuffer);                                     //освобождение командных буферов
}

void copyBuffer(VkApplication* app, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);                                                   //команда используется для копирования данных между двумя буферами
    endSingleTimeCommands(app,commandBuffer);
}

void generateMipmaps(VkApplication* app, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
{
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(app->getPhysicalDevice(), imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(commandBuffer,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit,
            VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    endSingleTimeCommands(app,commandBuffer);
}

void createImage(VkApplication* app, uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(app->getDevice(), &imageInfo, nullptr, &image) != VK_SUCCESS)
        throw std::runtime_error("failed to create image!");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(app->getDevice(), image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(app->getPhysicalDevice(), memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(app->getDevice(), &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate image memory!");

    vkBindImageMemory(app->getDevice(), image, imageMemory, 0);
}

void transitionImageLayout(VkApplication* app, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels)
{
    static_cast<void>(format);
    //Обработка переходов макета. Один из наиболее распространенных способов выполнения переходов макета - использование барьера памяти изображений
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;                                      //Первые два поля определяют переход макета.
    barrier.newLayout = newLayout;                                      //Можно использовать так, VK_IMAGE_LAYOUT_UNDEFINED как oldLayout будто вас не волнует существующее содержимое изображения.
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;              //Если вы используете барьер для передачи владения семейством очередей, то эти два поля должны быть индексами семейств очередей.
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;              //Они должны быть установлены на, VK_QUEUE_FAMILY_IGNORED если вы не хотите этого делать (не значение по умолчанию!).
    barrier.image = image;                                              //изображение, на которое влияют
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;    //и конкретную часть изображения
    barrier.subresourceRange.baseMipLevel = 0;                          //Наше изображение не является массивом,
    barrier.subresourceRange.levelCount = mipLevels;                    //поэтому указаны только один уровень и слой.
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    // Нам нужно обработать два перехода:
    //  * Не определено → пункт назначения передачи: передача пишет, что ничего не нужно ждать
    //  * Пункт назначения передачи → чтение шейдера: чтение шейдера должно ждать записи передачи,
    //   в частности, шейдер читает во фрагментном шейдере, потому что именно там мы собираемся использовать текстуру.*/

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }else{
        throw std::invalid_argument("unsupported layout transition!");
    }

    /* Барьеры в основном используются для целей синхронизации, поэтому вы должны указать, какие типы операций с ресурсом
     * должны выполняться до барьера, а какие операции, связанные с ресурсом, должны ждать на барьере.
     * Нам нужно это сделать, несмотря на то, что мы уже используем vkQueueWaitIdle синхронизацию вручную.
     * Правильные значения зависят от старого и нового макета, поэтому мы вернемся к этому, когда выясним,
     * какие переходы мы собираемся использовать.*/

    vkCmdPipelineBarrier(
        commandBuffer,                  //Первый параметр после буфера команд указывает, на каком этапе конвейера выполняются операции, которые должны произойти до барьера.
        sourceStage, destinationStage,  //Второй параметр указывает этап конвейера, на котором операции будут ожидать на барьере. Третий параметр - либо 0 или VK_DEPENDENCY_BY_REGION_BIT. Последнее превращает барьер в состояние для каждой области.
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(app, commandBuffer);
}

void copyBufferToImage(VkApplication* app, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);
    //Как и в случае с буферными копиями, вам нужно указать, какая часть буфера будет копироваться в какую часть изображения. Это происходит через VkBufferImageCopyструктуры:
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;                                     //определяют как пиксели раскладывают в памяти
    region.bufferImageHeight = 0;                                   //Указание 0для обоих означает, что пиксели просто плотно упакованы, как в нашем случае
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; //В imageSubresource, imageOffsetи imageExtent поля указывают , к какой части изображения мы хотим скопировать пиксели.
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {
        width,
        height,
        1
    };
    //Операции копирования из буфера в изображение ставятся в очередь с помощью vkCmdCopyBufferToImage функции
    //Четвертый параметр указывает, какой макет изображение используется в данный момент.
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(app, commandBuffer);
}

VkImageView createImageView(VkApplication* app, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;                                             //родительское изобраЖение для которого создаётся вид
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;                          //тип создаваемого вида, все типы на странице 73
    viewInfo.format = format;                                           //формат нового вида
    viewInfo.subresourceRange.aspectMask = aspectFlags;                 //является битовым полем состоящим из членов перечисления, задающим, на какие стороны изображения влияют барьеры
    viewInfo.subresourceRange.baseMipLevel = 0;                         //остальное не используется смотри, страницу 75
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

void createImageView(VkApplication* app, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels, VkImageView* imageView)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;                                             //родительское изобраЖение для которого создаётся вид
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;                          //тип создаваемого вида, все типы на странице 73
    viewInfo.format = format;                                           //формат нового вида
    viewInfo.subresourceRange.aspectMask = aspectFlags;                 //является битовым полем состоящим из членов перечисления, задающим, на какие стороны изображения влияют барьеры
    viewInfo.subresourceRange.baseMipLevel = 0;                         //остальное не используется смотри, страницу 75
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, imageView) != VK_SUCCESS)
    {throw std::runtime_error("failed to create texture image view!");}
}

void createCubeImage(VkApplication* app, uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 6;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(app->getDevice(), &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(app->getDevice(), image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(app->getPhysicalDevice(), memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(app->getDevice(), &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(app->getDevice(), image, imageMemory, 0);
}

VkImageView createCubeImageView(VkApplication* app, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;                                             //родительское изобраЖение для которого создаётся вид
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;                        //тип создаваемого вида, все типы на странице 73
    viewInfo.format = format;                                           //формат нового вида
    viewInfo.subresourceRange.aspectMask = aspectFlags;                 //является битовым полем состоящим из членов перечисления, задающим, на какие стороны изображения влияют барьеры
    viewInfo.subresourceRange.baseMipLevel = 0;                         //остальное не используется смотри, страницу 75
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;

    VkImageView imageView;
    if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

void transitionImageLayout(VkApplication* app, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t baseArrayLayer)
{
    static_cast<void>(format);
    //Обработка переходов макета. Один из наиболее распространенных способов выполнения переходов макета - использование барьера памяти изображений
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;                                      //Первые два поля определяют переход макета.
    barrier.newLayout = newLayout;                                      //Можно использовать так, VK_IMAGE_LAYOUT_UNDEFINED как oldLayout будто вас не волнует существующее содержимое изображения.
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;              //Если вы используете барьер для передачи владения семейством очередей, то эти два поля должны быть индексами семейств очередей.
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;              //Они должны быть установлены на, VK_QUEUE_FAMILY_IGNORED если вы не хотите этого делать (не значение по умолчанию!).
    barrier.image = image;                                              //изображение, на которое влияют
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;    //и конкретную часть изображения
    barrier.subresourceRange.baseMipLevel = 0;                          //Наше изображение не является массивом,
    barrier.subresourceRange.levelCount = mipLevels;                    //поэтому указаны только один уровень и слой.
    barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
    barrier.subresourceRange.layerCount = 6;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    // Нам нужно обработать два перехода:
    //  * Не определено → пункт назначения передачи: передача пишет, что ничего не нужно ждать
    //  * Пункт назначения передачи → чтение шейдера: чтение шейдера должно ждать записи передачи,
    //   в частности, шейдер читает во фрагментном шейдере, потому что именно там мы собираемся использовать текстуру.*/

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    /* Барьеры в основном используются для целей синхронизации, поэтому вы должны указать, какие типы операций с ресурсом
     * должны выполняться до барьера, а какие операции, связанные с ресурсом, должны ждать на барьере.
     * Нам нужно это сделать, несмотря на то, что мы уже используем vkQueueWaitIdle синхронизацию вручную.
     * Правильные значения зависят от старого и нового макета, поэтому мы вернемся к этому, когда выясним,
     * какие переходы мы собираемся использовать.*/

    vkCmdPipelineBarrier(
        commandBuffer,                  //Первый параметр после буфера команд указывает, на каком этапе конвейера выполняются операции, которые должны произойти до барьера.
        sourceStage, destinationStage,  //Второй параметр указывает этап конвейера, на котором операции будут ожидать на барьере. Третий параметр - либо 0 или VK_DEPENDENCY_BY_REGION_BIT. Последнее превращает барьер в состояние для каждой области.
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(app, commandBuffer);
}

void copyBufferToImage(VkApplication* app, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t baseArrayLayer)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);
    //Как и в случае с буферными копиями, вам нужно указать, какая часть буфера будет копироваться в какую часть изображения. Это происходит через VkBufferImageCopyструктуры:
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;                                     //определяют как пиксели раскладывают в памяти
    region.bufferImageHeight = 0;                                   //Указание 0для обоих означает, что пиксели просто плотно упакованы, как в нашем случае
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; //В imageSubresource, imageOffsetи imageExtent поля указывают , к какой части изображения мы хотим скопировать пиксели.
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = baseArrayLayer;
    region.imageSubresource.layerCount = 6;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {
        width,
        height,
        1
    };
    //Операции копирования из буфера в изображение ставятся в очередь с помощью vkCmdCopyBufferToImage функции
    //Четвертый параметр указывает, какой макет изображение используется в данный момент.
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(app, commandBuffer);
}

void generateMipmaps(VkApplication* app, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t baseArrayLayer)
{
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(app->getPhysicalDevice(), imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
    barrier.subresourceRange.layerCount = 6;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = baseArrayLayer;
        blit.srcSubresource.layerCount = 6;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = baseArrayLayer;
        blit.dstSubresource.layerCount = 6;

        vkCmdBlitImage(commandBuffer,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit,
            VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    endSingleTimeCommands(app,commandBuffer);
}

bool hasStencilComponent(VkFormat format)
{return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;}

VkFormat findSupportedFormat(VkApplication* app, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    VkFormat supportedFormat = candidates[0];   //VkFormat supportedFormat;
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(app->getPhysicalDevice(), format, &props);

        /* Структура VkFormatPropertiesсодержит три поля:
            * linearTilingFeatures: Варианты использования, которые поддерживаются линейной мозаикой.
            * optimalTilingFeatures: Варианты использования, которые поддерживаются оптимальной мозаикой.
            * bufferFeatures: Варианты использования, которые поддерживаются для буферов.
         * Здесь актуальны только первые два, а тот, который мы проверяем, зависит от tiling параметра функции:*/

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            supportedFormat = format;
            break;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            supportedFormat = format;
            break;
        }
        throw std::runtime_error("failed to find supported format!");
    }
    return supportedFormat;
}

VkFormat findDepthFormat(VkApplication* app)
{
    return findSupportedFormat(
        app,
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

VkFormat findDepthStencilFormat(VkApplication* app)
{
    return findSupportedFormat(
        app,
        {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    //Преимущество начала чтения в конце файла заключается в том,
    //что мы можем использовать позицию чтения для определения размера файла и выделения буфера:
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    //После этого мы можем вернуться к началу файла и прочитать все байты сразу:
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}
VkShaderModule createShaderModule(VkApplication* app, const std::vector<char>& code)
{
    //информация о шейдерном модуле которую мы переддим в vkCreateShaderModule в конце
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();                                  //размер модуля SPIR-V
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());  //передаётся сам код

    VkShaderModule shaderModule;    //дескриптор шейдерного модуля в который мы его создадим
    if (vkCreateShaderModule(app->getDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) //классическое дял вулкана создание шейдерного модуля
        throw std::runtime_error("failed to create shader module!");

    return shaderModule;
}

std::vector<QueueFamilyIndices> findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    std::vector<QueueFamilyIndices> indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int currentIndex = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        QueueFamilyIndices currentIndices;
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)                                         //побитное сравнение
            currentIndices.graphicsFamily = currentIndex;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, currentIndex, surface, &presentSupport);       //Запрос на поддержку презентации
        if (presentSupport)                                                                         //если поверхность и устройство поддерживают в i-ом семействе очередей,
            currentIndices.presentFamily = currentIndex;                                            //функция записывает в presentSupport значение true

        if (currentIndices.isComplete())                                                            //если оба значения не пусты, то функция вернёт true
            indices.push_back(currentIndices);

        currentIndex++;
    }

    return indices;
}

VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}

void outDeviceInfo(std::vector<physicalDevice>& physicalDevices)
{
    for(size_t i=0;i<physicalDevices.size();i++)
    {
        std::cout<<"Physical Device "<<i<<":"<<std::endl;
        for(size_t j=0;j<physicalDevices.at(i).indices.size();j++)
            std::cout<<"\tQueue Family Indices "<<physicalDevices.at(i).indices.at(j).presentFamily.value()<<std::endl;
    }
}

bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface, const std::vector<const char*>& deviceExtensions)
{
    bool extensionsSupported = checkDeviceExtensionSupport(device, deviceExtensions);

    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& deviceExtensions)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

    return requiredExtensions.empty();
}
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);  //запрашивает возможности поверхности

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);       //запрос количества поддерживаемых форматов поверхностей
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);                                                            //запись поддерживаемых форматов
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;      //аналогичный запрос поддерживамых режимов презентации
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}
