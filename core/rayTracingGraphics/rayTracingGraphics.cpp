#include "rayTracingGraphics.h"

#include "swapChain.h"
#include "vkdefault.h"

#include <cstring>

namespace moon::rayTracingGraphics {

RayTracingGraphics::RayTracingGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent)
    : shadersPath(shadersPath), workflowsShadersPath(workflowsShadersPath), extent(extent)
{
    setExtent(extent);
    link = &rayTracingLink;
}

RayTracingGraphics::~RayTracingGraphics(){
    RayTracingGraphics::destroy();
}

void RayTracingGraphics::ImageResource::create(const std::string& id, const moon::utils::PhysicalDevice& phDevice, const moon::utils::ImageInfo& imageInfo){
    this->id = id;

    host = new uint32_t[imageInfo.Extent.width * imageInfo.Extent.height];

    hostDevice.create(
        phDevice.instance,
        phDevice.getLogical(),
        sizeof(uint32_t) * imageInfo.Extent.width * imageInfo.Extent.height,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    device.create(
        phDevice.instance,
        phDevice.getLogical(),
        imageInfo,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

void RayTracingGraphics::ImageResource::destroy(const moon::utils::PhysicalDevice& phDevice){
    if(host){
        delete[] host;
        host = nullptr;
    }
}

void RayTracingGraphics::ImageResource::moveFromHostToHostDevice(VkExtent2D extent){
    hostDevice.copy(host);
}

void RayTracingGraphics::ImageResource::copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex){
    moon::utils::texture::transitionLayout(commandBuffer, device.image(imageIndex), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    moon::utils::texture::copy(commandBuffer, hostDevice, device.image(imageIndex), {extent.width, extent.height, 1}, 1);
    moon::utils::texture::transitionLayout(commandBuffer, device.image(imageIndex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
}

void RayTracingGraphics::create()
{
    CHECK(commandPool.create(device->getLogical()));

    emptyTexture = utils::Texture::empty(*device, commandPool);
    aDatabase.addEmptyTexture("black", &emptyTexture);

    moon::utils::ImageInfo imageInfo{ imageCount, format, extent, VK_SAMPLE_COUNT_1_BIT };

    color.create("color", *device, imageInfo);
    aDatabase.addAttachmentData(color.id, true, &color.device);

    bloom.create("bloom", *device, imageInfo);
    aDatabase.addAttachmentData(bloom.id, true, &bloom.device);

    moon::workflows::BloomParameters bloomParams;
    bloomParams.in.bloom = bloom.id;
    bloomParams.out.bloom = "finalBloom";

    bloomGraph = moon::workflows::BloomGraphics(imageInfo, workflowsShadersPath, bloomParams, bloomEnable, 8, VK_IMAGE_LAYOUT_UNDEFINED);
    bloomGraph.setDeviceProp(device->instance, device->getLogical());
    bloomGraph.create(aDatabase);
    bloomGraph.createCommandBuffers(commandPool);
    bloomGraph.updateDescriptorSets(bDatabase, aDatabase);

    moon::utils::ImageInfo bbInfo{imageCount, format, extent, VK_SAMPLE_COUNT_1_BIT};
    std::string bbId = "bb";
    bbGraphics.create(device->instance, device->getLogical(), bbInfo, shadersPath);
    aDatabase.addAttachmentData(bbId, bbGraphics.getEnable(), &bbGraphics.getAttachments());

    moon::utils::ImageInfo swapChainInfo{ imageCount, format, swapChainKHR->getExtent(), VK_SAMPLE_COUNT_1_BIT};
    RayTracingLinkParameters linkParams;
    linkParams.in.color = color.id;
    linkParams.in.bloom = bloomParams.out.bloom;
    linkParams.in.boundingBox = bbId;

    rayTracingLink.setParameters(linkParams);
    rayTracingLink.create(shadersPath, device->getLogical(), swapChainInfo);
    rayTracingLink.updateDescriptorSets(aDatabase);

    rayTracer.create();
}

void RayTracingGraphics::destroy() {
    color.destroy(*device);
    bloom.destroy(*device);
    aDatabase.destroy();
}

std::vector<std::vector<VkSemaphore>> RayTracingGraphics::submit(const std::vector<std::vector<VkSemaphore>>&, const std::vector<VkFence>&, uint32_t imageIndex)
{
    VkResult result = VK_SUCCESS;
    rayTracer.calculateImage(color.host, bloom.host);

    color.moveFromHostToHostDevice(extent);
    bloom.moveFromHostToHostDevice(extent);

    std::vector<VkCommandBuffer> commandBuffers;
    commandBuffers.push_back(moon::utils::singleCommandBuffer::create(device->getLogical(),commandPool));
    color.copyToDevice(commandBuffers.back(), extent, imageIndex);
    bloom.copyToDevice(commandBuffers.back(), extent, imageIndex);
    bbGraphics.render(commandBuffers.back(), imageIndex);
    moon::utils::singleCommandBuffer::submit(device->getLogical(), device->getQueue(0,0), commandPool, commandBuffers.size(), commandBuffers.data());

    bloomGraph.beginCommandBuffer(imageIndex);
    bloomGraph.updateCommandBuffer(imageIndex);
    bloomGraph.endCommandBuffer(imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &bloomGraph.getCommandBuffer(imageIndex);
    CHECK(result = vkQueueSubmit(device->getQueue(0,0), 1, &submitInfo, VK_NULL_HANDLE));
    CHECK(result = vkQueueWaitIdle(device->getQueue(0, 0)));

    return std::vector<std::vector<VkSemaphore>>();
}

void RayTracingGraphics::update(uint32_t imageIndex) {
    rayTracer.update();
    bbGraphics.update(imageIndex);
}

void RayTracingGraphics::setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size) {
    this->offset = offset;
    this->size = size;
    rayTracingLink.setPositionInWindow(offset, size);
}

void RayTracingGraphics::setEnableBoundingBox(bool enable){
    bbGraphics.setEnable(enable);
}

void RayTracingGraphics::setEnableBloom(bool enable){
    bloomEnable = enable;
}

void RayTracingGraphics::setBlitFactor(const float& blitFactor){
    bloomGraph.setBlitFactor(blitFactor);
}

void RayTracingGraphics::setExtent(VkExtent2D extent){
    this->extent = extent;
    rayTracer.setExtent(extent.width, extent.height);
}

void RayTracingGraphics::bind(cuda::rayTracing::Object* obj) {
    rayTracer.bind(obj);
}

void RayTracingGraphics::setCamera(cuda::rayTracing::Devicep<cuda::rayTracing::Camera>* cam){
    rayTracer.setCamera(cam);
    bbGraphics.bind(cam);
}

void RayTracingGraphics::clearFrame(){
    rayTracer.clearFrame();
}

void RayTracingGraphics::buildTree(){
    rayTracer.buildTree();
}

void RayTracingGraphics::buildBoundingBoxes(bool primitive, bool tree, bool onlyLeafs){
    bbGraphics.clear();

    if(tree){
        std::stack<cuda::rayTracing::KDNode<std::vector<const cuda::rayTracing::Primitive*>::iterator>*> stack;
        stack.push(rayTracer.getTree().getRoot());
        for(;!stack.empty();){
            const auto top = stack.top();
            stack.pop();

            if(!onlyLeafs || !(top->left || top->right)){
                std::random_device device;
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                cuda::rayTracing::cbox box(top->bbox, cuda::rayTracing::vec4f(dist(device), dist(device), dist(device), 1.0f));
                bbGraphics.bind(std::move(box));
            }

            if(top->right) stack.push(top->right);
            if(top->left) stack.push(top->left);
        }
    }

    if(primitive){
        for(auto& primitive: rayTracer.getTree().storage){
            cuda::rayTracing::cbox box(primitive->bbox, cuda::rayTracing::vec4f(1.0, 0.0, 0.0, 1.0f));
            bbGraphics.bind(box);
        }
    }
}
}
