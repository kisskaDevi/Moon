#ifndef CURSOR_H
#define CURSOR_H

#include <vulkan.h>
#include <vkdefault.h>
#include <buffer.h>

namespace moon::utils {

struct CursorPose {
    alignas(4) float x;
    alignas(4) float y;
};

struct CursorInfo {
    alignas(4) uint32_t number;
    alignas(4) float depth;
};

struct CursorBuffer {
    CursorPose pose;
    CursorInfo info;
};

class Cursor {
private:
    moon::utils::Buffer buffer;

public:
    void create(VkPhysicalDevice physicalDevice, VkDevice device);
    void update(const float& x, const float& y);
    CursorInfo read();

    VkDescriptorBufferInfo descriptorBufferInfo() const;
};

}
#endif // CURSOR_H
