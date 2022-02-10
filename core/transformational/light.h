#ifndef LIGHT_H
#define LIGHT_H

#include "core/vulkanCore.h"
#include "transformational.h"

class camera;
class shadowGraphics;
struct LightBufferObject;

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

template<typename type>
class light : public transformational {};

template<>
class light<spotLight> : public transformational
{
private:
    VkApplication                       *app;
    shadowGraphics                      *shadow;
    camera                              *camera;

    bool                                enableShadow = false;
    uint32_t                            type;
    uint32_t                            number;
    glm::vec4                           lightColor;

    glm::mat4x4                         projectionMatrix;
    glm::mat4x4                         viewMatrix;
    glm::mat4x4                         modelMatrix;

    glm::mat4x4                         m_globalTransform;
    glm::vec3                           m_translate;
    glm::vec3                           m_scale;
    glm::quat                           m_rotate;
    glm::quat                           m_rotateX;
    glm::quat                           m_rotateY;

    void updateViewMatrix();
    void renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet);
public:
    light(VkApplication *app, uint32_t imageCount, uint32_t type = lightType::spot);
    ~light();
    void cleanup();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void createLightPVM(const glm::mat4x4 & projection);

    void createShadow(uint32_t commandPoolsCount);

    void                            setImageCount(uint32_t imageCount);
    void                            setCamera(class camera *camera);
    void                            setLightColor(const glm::vec4 & color);
    void                            setLightNumber(const uint32_t & number);

    glm::mat4x4                     getViewMatrix() const;
    glm::mat4x4                     getModelMatrix() const;
    glm::vec3                       getTranslate() const;
    glm::vec4                       getLightColor() const;
    uint32_t                        getLightNumber() const;

    bool                            getShadowEnable() const;

    shadowGraphics                  *getShadow();

    LightBufferObject               getLightBufferObject() const;

};

template<>
class light<pointLight> : public transformational
{
private:
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
    light(VkApplication *app, uint32_t imageCount, std::vector<light<spotLight> *> & lightSource);
    ~light();

    void setLightColor(const glm::vec4 & color);
    void setCamera(class camera *camera);
    uint32_t getNumber() const;

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
