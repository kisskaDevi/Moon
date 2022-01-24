#ifndef LIGHT_H
#define LIGHT_H

#include "core/vulkanCore.h"
#include "transformational.h"

class camera;

class pointLight
{
private:
    int n = 6;
public:
    int getn(){ return n;}
};

class spotLight
{
private:
    int n = 1;
public:
    int getn()
    { return n;}
};

enum lightType
{
    spot,
    point
};

struct LightUniformBufferObject
{
    alignas(16) glm::mat4 projView;
    alignas(16) glm::vec4 position;
    alignas(16) glm::vec4 lightColor;
};

template<typename type>
class light : public transformational {};

template<>
class light<spotLight> : public transformational
{
private:
    VkApplication                                   *app;
    camera                                          *camera;

    bool                                            deleted = false;
    bool                                            enableShadow = false;
    int                                             LIGHT_COMMAND_POOLS;
    uint32_t                                        type;

    glm::mat4x4                                     projectionMatrix;
    glm::mat4x4                                     viewMatrix;
    glm::mat4x4                                     modelMatrix;
    glm::mat4x4                                     m_globalTransform;

    glm::vec4                                       lightColor;
    glm::vec3                                       m_translate;

    glm::vec3                                       m_scale;
    glm::quat                                       m_rotate;
    glm::quat                                       m_rotateX;
    glm::quat                                       m_rotateY;

    uint32_t                                        SHADOW_MAP_WIDTH = 1024;
    uint32_t                                        SHADOW_MAP_HEIGHT = 1024;
    uint32_t                                        mipLevels;
    uint32_t                                        imageCount;

    attachment                                      depthAttachment;
    VkSampler                                       shadowSampler;

    VkRenderPass                                    shadowRenerPass;
    std::vector<VkFramebuffer>                      shadowMapFramebuffer;

    VkDescriptorPool                                descriptorPool;
    std::vector<VkDescriptorSet>                    descriptorSets;
    VkPipelineLayout                                shadowPipelineLayout;
    VkDescriptorSetLayout                           descriptorSetLayout;
    VkDescriptorSetLayout                           uniformBufferSetLayout;
    VkDescriptorSetLayout                           uniformBlockSetLayout;
    VkPipeline                                      shadowPipeline;

    std::vector<VkCommandPool>                      shadowCommandPool;
    std::vector<std::vector<VkCommandBuffer>>       shadowCommandBuffer;

    std::vector<VkBuffer>                           lightUniformBuffers;
    std::vector<VkDeviceMemory>                     lightUniformBuffersMemory;

    void updateViewMatrix();
    void renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet);
public:
    light(VkApplication *app, uint32_t type = lightType::spot);
    ~light();

    void cleanup();
    void deleteLight();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void createShadowImage();
    void createShadowImageView();
    void createShadowSampler();
    void createLightPVM(const glm::mat4x4 & projection);

    void createShadowCommandPool();

    void createShadowRenderPass();
    void createShadowMapFramebuffer();

    void createShadowDescriptorSetLayout();
    void createShadowPipeline();

    void createShadowDescriptorPool();
    void createShadowDescriptorSets();

    void createShadowCommandBuffers(uint32_t number);
    void updateShadowCommandBuffers(uint32_t number, uint32_t i, std::vector<object *> & object3D);

    void createUniformBuffers();
    void updateUniformBuffer(uint32_t currentImage);

    void createShadow(uint32_t commandPoolsCount);

    void                            setImageCount(uint32_t imageCount);
    void                            setCamera(class camera *camera);
    void                            setLightColor(const glm::vec4 & color);
    void                            setCommandPoolsCount(const int & count);

    glm::mat4x4                     getViewMatrix() const;
    glm::mat4x4                     getModelMatrix() const;
    glm::vec3                       getTranslate() const;

    uint32_t                        getWidth() const;
    uint32_t                        getHeight() const;
    glm::vec4                       getLightColor() const;

    bool                            getShadowEnable() const;

    VkImage                         & getShadowImage();
    VkDeviceMemory                  & getShadowImageMemory();
    VkImageView                     & getImageView();
    VkSampler                       & getSampler();

    VkRenderPass                    & getShadowRenerPass();
    std::vector<VkFramebuffer>      & getShadowMapFramebuffer();
    VkPipeline                      & getShadowPipeline();
    VkCommandPool                   & getShadowCommandPool(uint32_t number);
    VkPipelineLayout                & getShadowPipelineLayout();
    std::vector<VkCommandBuffer>    & getCommandBuffer(uint32_t number);
    std::vector<VkBuffer>           & getLightUniformBuffers();
    std::vector<VkDeviceMemory>     & getLightUniformBuffersMemory();

};

template<>
class light<pointLight> : public transformational
{
private:
    bool deleted = false;

    glm::mat4 projectionMatrix;
    glm::vec3 m_translate;
    glm::quat m_rotate;
    glm::vec3 m_scale;
    glm::mat4x4 m_globalTransform;
    glm::quat m_rotateX;
    glm::quat m_rotateY;
    uint32_t number;
    glm::vec4 lightColor;

    std::vector<light<spotLight> *> & lightSource;

public:
    light(VkApplication *app, std::vector<light<spotLight> *> & lightSource);
    ~light();

    void setLightColor(const glm::vec4 & color);
    void setCamera(class camera *camera);
    uint32_t getNumber();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);
    void updateViewMatrix();

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    glm::vec3 getTranslate() const;
    glm::vec4 getLightColor() const;
};

#endif // LIGHT_H
