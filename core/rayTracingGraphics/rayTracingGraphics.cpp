#include "rayTracingGraphics.h"

#include "swapChain.h"
#include "vkdefault.h"

#include <cstring>

namespace moon::rayTracingGraphics {

RayTracingGraphics::RayTracingGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent)
    : shadersPath(shadersPath), workflowsShadersPath(workflowsShadersPath), extent(extent)
{
    setExtent(extent);
    Link.setShadersPath(shadersPath);
    link = &Link;
}

RayTracingGraphics::~RayTracingGraphics(){
    RayTracingGraphics::destroy();
    bbGraphics.destroy();
}

void RayTracingGraphics::ImageResource::create(const std::string& id, const moon::utils::PhysicalDevice& phDevice, VkFormat format, VkExtent2D extent, uint32_t imageCount){
    this->id = id;

    host = new uint32_t[extent.width * extent.height];

    moon::utils::buffer::create(
        phDevice.instance,
        phDevice.getLogical(),
        sizeof(uint32_t) * extent.width * extent.height,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &hostDevice.instance,
        &hostDevice.memory);
    vkMapMemory(phDevice.getLogical(), hostDevice.memory, 0, sizeof(uint32_t) * extent.width * extent.height, 0, &hostDevice.map);

    device.create(
        phDevice.instance,
        phDevice.getLogical(),
        format,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        extent,
        imageCount);
    VkSamplerCreateInfo SamplerInfo = moon::utils::vkDefault::samler();
    vkCreateSampler(phDevice.getLogical(), &SamplerInfo, nullptr, &device.sampler);
}

void RayTracingGraphics::ImageResource::destroy(const moon::utils::PhysicalDevice& phDevice){
    if(host){
        delete[] host;
        host = nullptr;
    }
    hostDevice.destroy(phDevice.getLogical());
}

void RayTracingGraphics::ImageResource::moveFromHostToHostDevice(VkExtent2D extent){
    std::memcpy(hostDevice.map, host, sizeof(uint32_t) * extent.width * extent.height);
}

void RayTracingGraphics::ImageResource::copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex){
    moon::utils::texture::transitionLayout(commandBuffer, device.instances[imageIndex].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    moon::utils::texture::copy(commandBuffer, hostDevice.instance, device.instances[imageIndex].image, {extent.width, extent.height, 1}, 1);
    moon::utils::texture::transitionLayout(commandBuffer, device.instances[imageIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
}

void RayTracingGraphics::create()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device->getLogical(), &poolInfo, nullptr, &commandPool);

    emptyTexture = createEmptyTexture(*device, commandPool);
    aDatabase.addEmptyTexture("black", emptyTexture);

    color.create("color", *device, format, extent, imageCount);
    aDatabase.addAttachmentData(color.id, true, &color.device);

    bloom.create("bloom", *device, format, extent, imageCount);
    aDatabase.addAttachmentData(bloom.id, true, &bloom.device);

    moon::utils::ImageInfo bloomInfo{imageCount, format, extent, VK_SAMPLE_COUNT_1_BIT};

    moon::workflows::BloomParameters bloomParams;
    bloomParams.in.bloom = bloom.id;
    bloomParams.out.bloom = "finalBloom";

    bloomGraph = new moon::workflows::BloomGraphics(bloomParams, bloomEnable, 8, VK_IMAGE_LAYOUT_UNDEFINED);
    bloomGraph->setShadersPath(workflowsShadersPath);
    bloomGraph->setDeviceProp(device->instance, device->getLogical());
    bloomGraph->setImageProp(&bloomInfo);
    bloomGraph->create(aDatabase);
    bloomGraph->createCommandBuffers(commandPool);
    bloomGraph->updateDescriptorSets(bDatabase, aDatabase);

    moon::utils::ImageInfo bbInfo{imageCount, format, extent, VK_SAMPLE_COUNT_1_BIT};
    std::string bbId = "bb";
    bbGraphics.create(device->instance, device->getLogical(), bbInfo, shadersPath);
    aDatabase.addAttachmentData(bbId, bbGraphics.getEnable(), &bbGraphics.getAttachments());

    moon::utils::ImageInfo swapChainInfo{ imageCount, format, swapChainKHR->getExtent(), VK_SAMPLE_COUNT_1_BIT};
    RayTracingLinkParameters linkParams;
    linkParams.in.color = color.id;
    linkParams.in.bloom = bloomParams.out.bloom;
    linkParams.in.boundingBox = bbId;

    Link.setParameters(linkParams);
    Link.setImageCount(imageCount);
    Link.setDeviceProp(device->getLogical());
    Link.setShadersPath(shadersPath);
    Link.createDescriptorSetLayout();
    Link.createPipeline(&swapChainInfo);
    Link.createDescriptorPool();
    Link.createDescriptorSets();
    Link.updateDescriptorSets(aDatabase);

    rayTracer.create();
}

void RayTracingGraphics::destroy() {
    if(emptyTexture){
        emptyTexture->destroy(device->getLogical());
        delete emptyTexture;
    }

    color.destroy(*device);
    bloom.destroy(*device);

    if(bloomGraph) delete bloomGraph;

    if(commandPool) {vkDestroyCommandPool(device->getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;}
    Link.destroy();
    bbGraphics.destroy();
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

    bloomGraph->beginCommandBuffer(imageIndex);
    bloomGraph->updateCommandBuffer(imageIndex);
    bloomGraph->endCommandBuffer(imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &bloomGraph->getCommandBuffer(imageIndex);
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
    Link.setPositionInWindow(offset, size);
}

void RayTracingGraphics::setEnableBoundingBox(bool enable){
    bbGraphics.setEnable(enable);
}

void RayTracingGraphics::setEnableBloom(bool enable){
    bloomEnable = enable;
}

void RayTracingGraphics::setBlitFactor(const float& blitFactor){
    bloomGraph->setBlitFactor(blitFactor);
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
