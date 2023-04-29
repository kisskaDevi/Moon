#include "buffer.h"

void buffer::destroy(VkDevice device){
    if(map){      vkUnmapMemory(device, memory); map = nullptr;}
    if(instance){ vkDestroyBuffer(device, instance, nullptr); instance = VK_NULL_HANDLE;}
    if(memory){   vkFreeMemory(device, memory, nullptr); memory = VK_NULL_HANDLE;}
}
