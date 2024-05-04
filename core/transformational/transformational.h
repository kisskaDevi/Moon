#ifndef TRANSFORMATIONAL_H
#define TRANSFORMATIONAL_H

#include "matrix.h"

namespace moon::transformational {

class Transformational
{
public:
    virtual ~Transformational(){};

    virtual Transformational& setGlobalTransform(const matrix<float,4,4>& transform) = 0;
    virtual Transformational& translate(const vector<float,3> & translate) = 0;
    virtual Transformational& rotate(const float & ang,const vector<float,3>& ax) = 0;
    virtual Transformational& scale(const vector<float,3>& scale) = 0;
};

}
#endif // TRANSFORMATIONAL_H
