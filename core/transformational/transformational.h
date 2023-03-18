#ifndef TRANSFORMATIONAL_H
#define TRANSFORMATIONAL_H

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <gtx/hash.hpp>

class transformational
{
public:
    virtual ~transformational(){};

    virtual void setGlobalTransform(const glm::mat4 & transform) = 0;
    virtual void translate(const glm::vec3 & translate) = 0;
    virtual void rotate(const float & ang,const glm::vec3 & ax) = 0;
    virtual void scale(const glm::vec3 & scale) = 0;
};

#endif // TRANSFORMATIONAL_H
