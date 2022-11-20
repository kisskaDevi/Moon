#include "camera.h"
#include "libs/dualQuaternion.h"

camera::camera(){}

camera::~camera(){}

void camera::updateViewMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);

    viewMatrix = glm::inverse(globalTransformation) * glm::inverse(transformMatrix);
}

void camera::setProjMatrix(const glm::mat4 & proj)
{
    projMatrix = proj;
}

void camera::setGlobalTransform(const glm::mat4 & transform)
{
    globalTransformation = transform;
    updateViewMatrix();
}

void camera::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateViewMatrix();
}

void camera::rotate(const float & ang ,const glm::vec3 & ax)
{
    rotation = convert(ang,ax)*rotation;
    updateViewMatrix();
}

void camera::scale(const glm::vec3 &scale)
{
    static_cast<void>(scale);
}

void camera::rotateX(const float & ang ,const glm::vec3 & ax)
{
    rotationX = convert(ang,ax) * rotationX;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void camera::rotateY(const float & ang ,const glm::vec3 & ax)
{
    rotationY = convert(ang,ax) * rotationY;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void camera::setPosition(const glm::vec3 & translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateViewMatrix();
}

void camera::setRotation(const float & ang ,const glm::vec3 & ax)
{
    rotation = convert(ang,ax);
    updateViewMatrix();
}

void    camera::setRotation(const quaternion<float>& rotation)
{
    this->rotation = rotation;
    updateViewMatrix();
}

void    camera::setRotations(const quaternion<float>& quatX, const quaternion<float>& quatY)
{
    this->rotationX = quatX;
    this->rotationY = quatY;
}

glm::mat4x4 camera::getProjMatrix() const
{
    return projMatrix;
}

glm::mat4x4 camera::getViewMatrix() const
{
    return viewMatrix;
}

glm::vec3           camera::getTranslation() const  {   return translation.vector();}
quaternion<float>   camera::getRotationX()const     {   return rotationX;}
quaternion<float>   camera::getRotationY()const     {   return rotationY;}
