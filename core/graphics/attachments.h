#ifndef ATTACHMENTS_H
#define ATTACHMENTS_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3native.h>
#include <vector>

class attachments
{
private:
    size_t size;
public:
    std::vector<VkImage> image;
    std::vector<VkDeviceMemory> imageMemory;
    std::vector<VkImageView> imageView;
    VkSampler sampler;

    attachments();
    ~attachments();

    void resize(size_t size);
    void deleteAttachment(VkDevice * device);
    void deleteSampler(VkDevice * device);
    void setSize(size_t size);
    size_t getSize();
};

#endif // ATTACHMENTS_H
