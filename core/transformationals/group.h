#ifndef GROUP_H
#define GROUP_H

#include <unordered_set>

#include "transformational.h"
#include "quaternion.h"

namespace moon::transformational {

class Group : public Transformational
{
private:
    std::unordered_set<Transformational*> objects;
    moon::math::Matrix<float, 4, 4> modelMatrix{1.0f};
    DEFAULT_TRANSFORMATIONAL()

public:
    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Group)
    DEFAULT_TRANSFORMATIONAL_GETTERS()

    void addObject(Transformational* object);
    void delObject(Transformational* object);
    bool findObject(Transformational* object);
};

}
#endif // GROUP_H
