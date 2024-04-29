#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "rayTracingLink.h"
#include "graphicsInterface.h"
#include "attachments.h"
#include "buffer.h"
#include "texture.h"
#include "vector.h"

#include <stdint.h>
#include <random>

#include "cudaRayTracing.h"
#include "boundingBoxGraphics.h"
#include <bloom.h>

namespace cuda { class model;}

class rayTracingGraphics : public graphicsInterface {
private:
    struct imageResource{
        std::string id;
        uint32_t* host{nullptr};
        buffer hostDevice;
        attachments device;

        void create(const std::string& id, physicalDevice phDevice, VkFormat format, VkExtent2D extent, uint32_t imageCount);
        void destroy(physicalDevice phDevice);
        void moveFromHostToHostDevice(VkExtent2D extent);
        void copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex);
    };

    imageResource color;
    imageResource bloom;

    cuda::cudaRayTracing rayTracer;
    boundingBoxGraphics bbGraphics;
    bloomGraphics bloomGraph;
    rayTracingLink Link;

    texture* emptyTexture{nullptr};

    std::filesystem::path shadersPath;
    std::filesystem::path workflowsShadersPath;
    VkExtent2D extent;

    attachmentsDatabase aDatabase;
    buffersDatabase bDatabase;

    VkCommandPool commandPool{VK_NULL_HANDLE};

    bool bloomEnable = true;

public:
    rayTracingGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent)
        : shadersPath(shadersPath), workflowsShadersPath(workflowsShadersPath), extent(extent)
    {
        setExtent(extent);
        Link.setShadersPath(shadersPath);
        link = &Link;
    }

    void setPositionInWindow(const vector<float,2>& offset, const vector<float,2>& size) override {
        this->offset = offset;
        this->size = size;
        Link.setPositionInWindow(offset, size);
    }

    ~rayTracingGraphics(){
        rayTracingGraphics::destroy();
        bbGraphics.destroy();
    }

    void setEnableBoundingBox(bool enable){
        bbGraphics.setEnable(enable);
    }

    void setEnableBloom(bool enable){
        bloomEnable = enable;
    }

    void setBlitFactor(const float& blitFactor){
        bloomGraph.setBlitFactor(blitFactor);
    }

    void setExtent(VkExtent2D extent){
        this->extent = extent;
        rayTracer.setExtent(extent.width, extent.height);
    }

    void bind(cuda::model* m) {
        rayTracer.bind(m);
        for(const auto& primitive: m->primitives){
            //bbGraphics.bind(primitive.box);
        }
    }

    void setCamera(cuda::devicep<cuda::camera>* cam){
        rayTracer.setCamera(cam);
        bbGraphics.bind(cam);
    }

    void clearFrame(){
        rayTracer.clearFrame();
    }


    void create() override;
    void destroy() override;
    void update(uint32_t imageIndex) override;
    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    void bindNextNode(cuda::cudaRayTracing::kdTree_host* node, size_t maxDepth, size_t& depth){
        depth++;
        std::random_device device;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        node->box.color = cuda::vec4f(dist(device), dist(device), dist(device), 1.0f);
        bbGraphics.bind(node->box);
        if(node->left){
            bindNextNode(node->left, maxDepth, depth);
        }
        if(node->right){
            bindNextNode(node->right, maxDepth, depth);
        }
        depth--;
    }
    void buildTree(){
        rayTracer.buildTree();
        size_t maxDepth;
        size_t depth = 0;
        const auto root = rayTracer.getTree(maxDepth);
        bindNextNode(root, maxDepth, depth);
    }
};

#endif // !RAYTRACINGGRAPHICS

