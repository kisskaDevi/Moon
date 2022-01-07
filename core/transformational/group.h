#ifndef GROUP_H
#define GROUP_H

#include "transformational.h"

class group : public transformational
{
private:
    std::vector<transformational *> objects;
    glm::vec3 m_translate;
    glm::quat m_rotate;
    glm::vec3 m_scale;
    glm::mat4x4 m_globalTransform;

public:
    group();
    ~group();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang ,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void addObject(transformational* obj);
    void delObject(const int &index);
};

#endif // GROUP_H
