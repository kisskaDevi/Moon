#ifndef GROUP_H
#define GROUP_H

#include <vector>

#include "transformational.h"
#include "quaternion.h"

namespace moon::transformational {

class Group : public Transformational
{
private:
    std::vector<Transformational *>     objects;

    moon::math::Quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    moon::math::Vector<float,3>         scaling{1.0f,1.0f,1.0f};
    moon::math::Matrix<float,4,4>       globalTransformation{1.0f};
    moon::math::Matrix<float,4,4>       modelMatrix{1.0f};

    void updateModelMatrix();

public:
    Group& setGlobalTransform(const moon::math::Matrix<float,4,4> & transform);
    Group& translate(const moon::math::Vector<float,3> & translate);
    Group& rotate(const float & ang ,const moon::math::Vector<float,3> & ax);
    Group& scale(const moon::math::Vector<float,3> & scale);

    void addObject(Transformational* object);
    void delObject(Transformational* object);
    bool findObject(Transformational* object);
};

}
#endif // GROUP_H
