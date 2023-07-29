#ifndef GROUP_H
#define GROUP_H

#include <vector>

#include "transformational.h"
#include "quaternion.h"

class group : public transformational
{
private:
    std::vector<transformational *>     objects;

    quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    vector<float,3>         scaling{1.0f,1.0f,1.0f};
    matrix<float,4,4>       globalTransformation{1.0f};
    matrix<float,4,4>       modelMatrix{1.0f};

    void updateModelMatrix();
public:
    group();
    ~group();

    void setGlobalTransform(const matrix<float,4,4> & transform);
    void translate(const vector<float,3> & translate);
    void rotate(const float & ang ,const vector<float,3> & ax);
    void scale(const vector<float,3> & scale);

    void addObject(transformational* object);
    void delObject(const int& index);
};

#endif // GROUP_H
