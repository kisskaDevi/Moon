#ifndef PHYSICALOBJECT_H
#define PHYSICALOBJECT_H

#include "core/transformational/object.h"

class physicalObject : public object
{
public:
    physicalObject(VkApplication* app, gltfModel* model3D, float mass);
    physicalObject(VkApplication* app, objectInfo info, float mass);

    void setAcceleration(const glm::vec3& acceleration, const float& dt);
    void setVelocity(const glm::vec3& velocity);
    void setCollisionRadiusСoefficient(const float& R);
    void setEnergy(const float& E);
    void setAngularVelocity(const glm::vec3& w);

    void setResistanceFactor(const float& r);
    void setAngularResistanceFactor(const float& ar);

    glm::vec3   getVelocity() const;
    glm::vec3   getAngularVelocity() const;
    float       getMass() const;
    float       getEnergy() const;
    float       getCollisionRadiusСoefficient() const;
    float       getSurfaceFriction() const;
    bool        isPhysicsEnable() const;

    void update(const float& dt);
    void setPhysics(const bool& enable);

    friend bool findCollision(physicalObject* object1, physicalObject* object2, const float& dt);
    friend bool findSphCollision(physicalObject* object1, physicalObject* object2, const float& dt);
    friend void sphCollision(physicalObject* object1, physicalObject* object2, glm::vec3& O1, glm::vec3& O2, float& R1, float& R2, const float& dt);
    friend void collision(physicalObject* object1, physicalObject* object2, glm::vec3& O1, glm::vec3& O2, const float& dt);

    friend bool FindCollision1(physicalObject* object1, physicalObject* object2, const float& dt);
    friend bool FindCollision2(physicalObject* object1, glm::vec4& min1, glm::vec4& max1, glm::mat4& mat1, glm::vec3& O1, physicalObject* object2, glm::vec4& min2, glm::vec4& max2, glm::mat4& mat2, glm::vec3& O2, const float& dt);
    friend bool findCollision1(physicalObject* object1, physicalObject* object2, const float& dt);
    friend bool findCollision2(physicalObject* object1, glm::vec4& min1, glm::vec4& max1, glm::mat4& mat1, glm::vec3& O1, physicalObject* object2, glm::vec4& min2, glm::vec4& max2, glm::mat4& mat2, glm::vec3& O2, const float& dt);

    friend bool FindCollision1(object* object1, physicalObject* object2, const float& dt);
    friend bool FindCollision2(object* object1, glm::vec4& min1, glm::vec4& max1, glm::mat4& mat1, glm::vec3& O1, physicalObject* object2, glm::vec4& min2, glm::vec4& max2, glm::mat4& mat2, glm::vec3& O2, const float& dt);

private:
    float                           resistanceFactor = 0.0f;
    float                           angularResistanceFactor = 0.0f;

    float                           collisionRadiusСoefficient = 0.0f;
    float                           mass;
    float                           surfaceFriction = 0.1f;

    glm::vec3                       velocity = {0.0f,0.0f,0.0f};
    glm::vec3                       acceleration = {0.0f,0.0f,0.0f};

    glm::vec3                       angularVelocity = {0.0f,0.0f,0.0f};
    glm::vec3                       angularAcceleration = {0.0f,0.0f,0.0f};

    float                           energy;

    bool                            enable = true;
};

void sortMax(glm::vec4 &min, glm::vec4 &max);

glm::vec3 findMin(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3);
glm::vec3 findMax(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3);

#endif // PHYSICALOBJECT_H
