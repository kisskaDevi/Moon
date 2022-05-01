#include "camera.h"

camera::camera()
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    viewMatrix = glm::mat4x4(1.0f);
}

camera::~camera()
{

}

void camera::setPosition(const glm::vec3 & translate)
{
    m_translate = translate;
    updateViewMatrix();
}

void camera::setRotation(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax));
    updateViewMatrix();
}

void camera::defaultPosition()
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    viewMatrix = glm::mat4x4(1.0f);
}

void camera::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void camera::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void camera::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void camera::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void camera::updateViewMatrix()
{
    glm::mat4x4 scaleMatrix = glm::scale(glm::mat4x4(1.0f),m_scale);
    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
    {
        rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    }
    glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),-m_translate);
    viewMatrix = glm::inverse(m_globalTransform) * scaleMatrix * rotateMatrix * translateMatrix;
}

glm::mat4x4 camera::getViewMatrix() const
{
    return viewMatrix;
}

glm::vec3 camera::getTranslate() const
{
    return m_translate;
}

void camera::rotateX(const float & ang ,const glm::vec3 & ax)
{
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void camera::rotateY(const float & ang ,const glm::vec3 & ax)
{
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}
