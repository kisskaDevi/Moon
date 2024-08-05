#include "group.h"
#include "dualQuaternion.h"

namespace moon::transformational {

Group& Group::update()
{
    moon::math::Matrix<float,4,4> transformMatrix = convert(convert(m_rotation, m_translation));
    modelMatrix = m_globalTransformation * transformMatrix * moon::math::scale(m_scaling);
    for (auto& object : objects) {
        object->setGlobalTransform(modelMatrix);
    }
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Group)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Group)

void Group::addObject(Transformational* object)
{
    if (auto [_, inserted] = objects.insert(object); inserted) {
        update();
    }
}

void Group::delObject(Transformational* object) {
    objects.erase(object);
}

bool Group::findObject(Transformational* object) {
    return objects.find(object) != objects.end();
}

}
