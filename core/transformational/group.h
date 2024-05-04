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

    quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    vector<float,3>         scaling{1.0f,1.0f,1.0f};
    matrix<float,4,4>       globalTransformation{1.0f};
    matrix<float,4,4>       modelMatrix{1.0f};

    void updateModelMatrix();
public:
    Group();
    ~Group();

    Group& setGlobalTransform(const matrix<float,4,4> & transform);
    Group& translate(const vector<float,3> & translate);
    Group& rotate(const float & ang ,const vector<float,3> & ax);
    Group& scale(const vector<float,3> & scale);

    void addObject(Transformational* object);
    void delObject(Transformational* object);
    bool findObject(Transformational* object);
};

}
#endif // GROUP_H
