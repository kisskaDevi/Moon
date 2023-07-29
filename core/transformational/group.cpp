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

    modelMatrix = globalTransformation * transformMatrix * ::scale(scaling);
}

void group::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang, vector<float,3>(normalize(ax)))*rotation;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::scale(const vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
}

void group::setGlobalTransform(const matrix<float,4,4> & transform)
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
