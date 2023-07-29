#ifndef TRANSFORMATIONAL_H
#define TRANSFORMATIONAL_H

#include "matrix.h"

class transformational
{
public:
    virtual ~transformational(){};

    virtual void setGlobalTransform(const matrix<float,4,4>& transform) = 0;
    virtual void translate(const vector<float,3> & translate) = 0;
    virtual void rotate(const float & ang,const vector<float,3>& ax) = 0;
    virtual void scale(const vector<float,3>& scale) = 0;
};

#endif // TRANSFORMATIONAL_H
