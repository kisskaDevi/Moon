#include "group.h"
#include "dualQuaternion.h"

group::group()
{}

group::~group()
{}

void group::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    modelMatrix = globalTransformation * transformMatrix * scaleMatrix;
}

void group::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang, ax)*rotation;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::scale(const glm::vec3 & scale)
{
    scaling = scale;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::setGlobalTransform(const glm::mat4 & transform)
{
    globalTransformation = transform;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::addObject(transformational* object)
{
    objects.push_back(object);
    updateModelMatrix();

    object->setGlobalTransform(modelMatrix);
}

void group::delObject(const int &index)
{
    objects.erase(objects.begin()+index);
}
