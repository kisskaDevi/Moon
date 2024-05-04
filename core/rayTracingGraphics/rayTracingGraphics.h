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
#include <stack>

#include "cudaRayTracing.h"
#include "boundingBoxGraphics.h"
#include <bloom.h>

namespace cuda { class model;}

class rayTracingGraphics : public moon::graphicsManager::GraphicsInterface {
private:
    struct imageResource{
        std::string id;
        uint32_t* host{nullptr};
        moon::utils::Buffer hostDevice;
        moon::utils::Attachments device;

        void create(const std::string& id, moon::utils::PhysicalDevice phDevice, VkFormat format, VkExtent2D extent, uint32_t imageCount);
        void destroy(moon::utils::PhysicalDevice phDevice);
        void moveFromHostToHostDevice(VkExtent2D extent);
        void copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex);
    };

    imageResource color;
    imageResource bloom;

    cuda::cudaRayTracing rayTracer;
    boundingBoxGraphics bbGraphics;
    moon::workflows::BloomGraphics bloomGraph;
    rayTracingLink Link;

    moon::utils::Texture* emptyTexture{nullptr};

    std::filesystem::path shadersPath;
    std::filesystem::path workflowsShadersPath;
    VkExtent2D extent;

    moon::utils::AttachmentsDatabase aDatabase;
    moon::utils::BuffersDatabase bDatabase;

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

    void buildTree(){
        rayTracer.buildTree();
    }

    void buildBoundingBoxes(bool primitive, bool tree, bool onlyLeafs){
        bbGraphics.clear();

        if(tree){
            std::stack<cuda::kdNode<std::vector<const cuda::primitive*>::iterator>*> stack;
            stack.push(rayTracer.getTree().getRoot());
            for(;!stack.empty();){
                const auto top = stack.top();
                stack.pop();

                if(!onlyLeafs || !(top->left || top->right)){
                    std::random_device device;
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    cuda::cbox box(top->bbox, cuda::vec4f(dist(device), dist(device), dist(device), 1.0f));
                    bbGraphics.bind(box);
                }

                if(top->right) stack.push(top->right);
                if(top->left) stack.push(top->left);
            }
        }

        if(primitive){
            for(auto& primitive: rayTracer.getTree().storage){
                std::random_device device;
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                cuda::cbox box(primitive->bbox, cuda::vec4f(1.0, 0.0, 0.0, 1.0f));
                bbGraphics.bind(box);
            }
        }
    }
};

#endif // !RAYTRACINGGRAPHICS

