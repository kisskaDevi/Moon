#ifndef GROUP_H
#define GROUP_H

#include "transformational.h"
#include "quaternion.h"

class group : public transformational
{
private:
    std::vector<transformational *>     objects;

    quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    glm::vec3               scaling{1.0f,1.0f,1.0f};
    glm::mat4x4             globalTransformation{1.0f};
    glm::mat4x4             modelMatrix{1.0f};

    void updateModelMatrix();
public:
    group();
    ~group();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang ,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void addObject(transformational* object);
    void delObject(const int& index);
};

#endif // GROUP_H
