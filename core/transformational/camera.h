#ifndef CAMERA_H
#define CAMERA_H

#include "transformational.h"

class camera : public transformational
{
private:
    glm::mat4x4 viewMatrix;
    glm::vec3 m_translate;
    glm::quat m_rotate;
    glm::vec3 m_scale;
    glm::mat4x4 m_globalTransform;
    glm::quat m_rotateX;
    glm::quat m_rotateY;

public:
    camera();
    ~camera();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang ,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void updateViewMatrix();
    glm::mat4x4 getViewMatrix() const;
    glm::vec3 getTranslate() const;

};

#endif // CAMERA_H
