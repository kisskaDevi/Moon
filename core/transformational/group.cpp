#include "group.h"
#include "dualQuaternion.h"

group::group()
{}

group::~group()
{}

void group::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    matrix<float,4,4> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    glm::mat<4,4,float,glm::defaultp> glmTransformMatrix;
    for(uint32_t i=0;i<4;i++){
        for(uint32_t j=0;j<4;j++){
            glmTransformMatrix[i][j] = transformMatrix[i][j];
        }
    }

    modelMatrix = globalTransformation * glmTransformMatrix * scaleMatrix;
}

void group::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang, vector<float,3>(ax[0],ax[1],ax[2]))*rotation;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,vector<float,3>(translate[0],translate[1],translate[2]));
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
