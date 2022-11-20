#ifndef CAMERA_H
#define CAMERA_H

#include "transformational.h"
#include "libs/quaternion.h"

class camera : public transformational
{
private:
    glm::mat4x4         projMatrix{1.0f};
    glm::mat4x4         viewMatrix{1.0f};
    glm::mat4x4         globalTransformation{1.0f};

    quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotationX{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotationY{1.0f,0.0f,0.0f,0.0f};

    void updateViewMatrix();
public:
    camera();
    ~camera();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang ,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void                    setProjMatrix(const glm::mat4 & proj);
    void                    setPosition(const glm::vec3 & translate);
    void                    setRotation(const float & ang ,const glm::vec3 & ax);
    void                    setRotation(const quaternion<float>& rotation);
    void                    setRotations(const quaternion<float>& rotationX, const quaternion<float>& rotationY);

    glm::vec3               getTranslation()const;
    quaternion<float>       getRotationX()const;
    quaternion<float>       getRotationY()const;

    glm::mat4x4             getProjMatrix() const;
    glm::mat4x4             getViewMatrix() const;
};

#endif // CAMERA_H
