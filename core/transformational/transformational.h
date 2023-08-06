#ifndef TRANSFORMATIONAL_H
#define TRANSFORMATIONAL_H

#include "matrix.h"

class transformational
{
public:
    virtual ~transformational(){};

    virtual transformational& setGlobalTransform(const matrix<float,4,4>& transform) = 0;
    virtual transformational& translate(const vector<float,3> & translate) = 0;
    virtual transformational& rotate(const float & ang,const vector<float,3>& ax) = 0;
    virtual transformational& scale(const vector<float,3>& scale) = 0;
};

#endif // TRANSFORMATIONAL_H
