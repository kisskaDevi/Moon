#include "group.h"
#include "dualQuaternion.h"

#include <numeric>

namespace moon::transformational {

Group::Group(){}
Group::~Group(){}

void Group::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * ::scale(scaling);
}

Group& Group::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang, vector<float,3>(normalize(ax)))*rotation;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
    return *this;
}

Group& Group::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
    return *this;
}

Group& Group::scale(const vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
    return *this;
}

Group& Group::setGlobalTransform(const matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();

    for(auto& object: objects){
        object->setGlobalTransform(modelMatrix);
    }
    return *this;
}

void Group::addObject(Transformational* object)
{
    objects.push_back(object);
    updateModelMatrix();

    object->setGlobalTransform(modelMatrix);
}

void Group::delObject(Transformational* object)
{
    for(auto objIt = objects.begin(); objIt != objects.end(); objIt++){
        if(*objIt == object){
            objects.erase(objIt);
        }
    }
}

bool Group::findObject(Transformational* object)
{
    return std::accumulate(objects.begin(),objects.end(), false, [&object](const bool& a, const auto& obj){
        return a + (obj == object);
    });
}

}
