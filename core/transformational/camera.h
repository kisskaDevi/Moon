#ifndef CAMERA_H
#define CAMERA_H

#include "transformational.h"

#include "libs/dualQuaternion.h"

class camera : public transformational
{
private:
    glm::mat4x4         projMatrix;
    glm::mat4x4         viewMatrix;
    glm::vec3           m_translate;
    glm::quat           m_rotate;
    glm::vec3           m_scale;
    glm::mat4x4         m_globalTransform;
    glm::quat           m_rotateX;
    glm::quat           m_rotateY;

    quaternion<float>   quat;
    quaternion<float>   quatX;
    quaternion<float>   quatY;
    dualQuaternion<float>   dQuat;

public:
    camera();
    ~camera();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang ,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void setPosition(const glm::vec3 & translate);
    void setRotation(const float & ang ,const glm::vec3 & ax);

    void setProjMatrix(const glm::mat4 & proj);
    glm::mat4x4 getProjMatrix() const;

    void updateViewMatrix();
    glm::mat4x4 getViewMatrix() const;
    glm::vec3 getTranslate() const;

    void                    setDualQuaternion(const dualQuaternion<float>& dQuat);
    dualQuaternion<float>   getDualQuaternion()const;

    void                    setQuaternion(const quaternion<float>& quat);
    void                    setQuaternions(const quaternion<float>& quatX, const quaternion<float>& quatY);
    quaternion<float>       getquatX()const;
    quaternion<float>       getquatY()const;

};

#endif // CAMERA_H
