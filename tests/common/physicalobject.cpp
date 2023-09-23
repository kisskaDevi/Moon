#include "physicalobject.h"
#include "../../core/models/gltfmodel.h"

physicalObject::physicalObject(VkApplication* app, gltfModel* model3D, float mass) : object(app, model3D)
{
    this->mass = mass;
    velocity = glm::vec3(0.0f,0.0f,0.0f);
    acceleration = glm::vec3(0.0f,0.0f,0.0f);
}

physicalObject::physicalObject(VkApplication* app, objectInfo info, float mass) : object(app, info)
{
    this->mass = mass;
    velocity = glm::vec3(0.0f,0.0f,0.0f);
    acceleration = glm::vec3(0.0f,0.0f,0.0f);
}

void physicalObject::setAcceleration(const glm::vec3& acceleration, const float& dt)
{
    this->acceleration = acceleration;
    this->velocity += acceleration * dt;
}

void physicalObject::setVelocity(const glm::vec3& velocity)         {   this->velocity = velocity;}
void physicalObject::setCollisionRadiusСoefficient(const float& R)  {   collisionRadiusСoefficient = R;}
void physicalObject::setEnergy(const float& E)                      {   energy = E;}
void physicalObject::setAngularVelocity(const glm::vec3& w)         {   angularVelocity = w;}
void physicalObject::setResistanceFactor(const float& r)            {   resistanceFactor = r;}
void physicalObject::setAngularResistanceFactor(const float& ar)    {   angularResistanceFactor = ar;}
void physicalObject::setPhysics(const bool& enable)                 {   this->enable = enable;}

glm::vec3   physicalObject::getVelocity() const                     {   return velocity;}
glm::vec3   physicalObject::getAngularVelocity() const              {   return angularVelocity;}
float       physicalObject::getMass() const                         {   return mass;}
float       physicalObject::getEnergy() const                       {   return energy;}
float       physicalObject::getCollisionRadiusСoefficient() const   {   return collisionRadiusСoefficient;}
float       physicalObject::getSurfaceFriction() const              {   return surfaceFriction;}
bool        physicalObject::isPhysicsEnable() const                 {   return enable;}

void physicalObject::update(const float& dt)
{
        glm::vec3 resistance = -resistanceFactor*velocity;
        this->velocity += resistance * dt;

        glm::vec3 angularResistance = -angularResistanceFactor*angularVelocity;
        this->angularVelocity += angularResistance * dt;
        glm::vec3 ds = velocity * dt;
        object::translate(ds);

        float w = glm::length(angularVelocity) * dt;
        if(w!=0.0f&&!(angularVelocity.x==0.0f&&angularVelocity.y==0.0f&&angularVelocity.z==0.0f)){
            object::rotate(w,glm::normalize(angularVelocity));
        }
}

void sortMax(glm::vec4 &min, glm::vec4 &max)
{
    glm::vec4 localmin = min;
    glm::vec4 localmax = max;

    if(localmin.x>localmax.x){
        max.x = localmin.x;
        min.x = localmax.x;
    }
    if(localmin.y>localmax.y){
        max.y = localmin.y;
        min.y = localmax.y;
    }
    if(localmin.z>localmax.z){
        max.z = localmin.z;
        min.z = localmax.z;
    }
}

bool findCollision(physicalObject* object1, physicalObject* object2, const float& dt)
{
    bool result = false;
    for(auto *node1: object1->getModel()->linearNodes){
        if (node1->mesh) {
            glm::mat4x4 mat1 = object1->ModelMatrix() * node1->getMatrix();
            for(auto *primitive1: node1->mesh->primitives){
                glm::vec4 max1 = mat1 * glm::vec4(primitive1->bb.max,1.0f);
                glm::vec4 min1 = mat1 * glm::vec4(primitive1->bb.min,1.0f);
                sortMax(min1,max1);
                glm::vec3 O1 = (glm::vec3(max1) + glm::vec3(min1))/2.0f;
                for(auto *node2: object2->getModel()->linearNodes){
                    if (node2->mesh) {
                        glm::mat4x4 mat2 = object2->ModelMatrix() * node2->getMatrix();
                        for(auto *primitive2: node2->mesh->primitives){
                            glm::vec4 max2 = mat2 * glm::vec4(primitive2->bb.max,1.0f);
                            glm::vec4 min2 = mat2 * glm::vec4(primitive2->bb.min,1.0f);
                            sortMax(min2,max2);
                            glm::vec3 O2 = (glm::vec3(max2) + glm::vec3(min2))/2.0f;

                            bool xcollision = max1.x>=min2.x&&max2.x>=min1.x;
                            bool ycollision = max1.y>=min2.y&&max2.y>=min1.y;
                            bool zcollision = max1.z>=min2.z&&max2.z>=min1.z;
                            result = xcollision&&ycollision&&zcollision;

                            if(result){
                                collision(object1, object2, O1, O2, dt);
                                return result;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

void collision(physicalObject* object1, physicalObject* object2, glm::vec3& O1, glm::vec3& O2, const float& dt)
{
    glm::vec3 v1 = object1->getVelocity();
    glm::vec3 v2 = object2->getVelocity();
    glm::vec3 n  = glm::normalize(O2-O1);
    float     e  = 1.0f;
    float     m1 = object1->getMass();
    float     m2 = object2->getMass();

    glm::vec3 un1 = ((1+e)*m2*glm::dot(v2,n) + (m1-e*m2)*glm::dot(v1,n))/(m1+m2) * n;
    glm::vec3 un2 = ((1+e)*m1*glm::dot(v1,n) + (m2-e*m1)*glm::dot(v2,n))/(m1+m2) * n;

    glm::vec3 ut1 = v1 - glm::dot(n,v1) * n;
    glm::vec3 ut2 = v2 - glm::dot(n,v2) * n;

    glm::vec3 u1 = un1 + ut1;
    glm::vec3 u2 = un2 + ut2;

    object1->setVelocity(u1);
    object2->setVelocity(u2);

//    object1->update(dt);
//    object2->update(dt);

//    findCollision(object1, object2, dt);
}

bool findSphCollision(physicalObject* object1, physicalObject* object2, const float& dt)
{
    bool result = false;
    for(auto *node1: object1->getModel()->linearNodes){
        if (node1->mesh) {
            glm::mat4x4 mat1 = object1->ModelMatrix() * node1->getMatrix();
            for(auto *primitive1: node1->mesh->primitives){
                glm::vec4 max1 = mat1 * glm::vec4(primitive1->bb.max,1.0f);
                glm::vec4 min1 = mat1 * glm::vec4(primitive1->bb.min,1.0f);
                glm::vec3 O1 = (glm::vec3(max1) + glm::vec3(min1))/2.0f;
                float R1 = glm::length(O1-glm::vec3(max1)) * object1->collisionRadiusСoefficient;
                for(auto *node2: object2->getModel()->linearNodes){
                    if (node2->mesh) {
                        glm::mat4x4 mat2 = object2->ModelMatrix() * node2->getMatrix();
                        for(auto *primitive2: node2->mesh->primitives){
                            glm::vec4 max2 = mat2 * glm::vec4(primitive2->bb.max,1.0f);
                            glm::vec4 min2 = mat2 * glm::vec4(primitive2->bb.min,1.0f);
                            //sortMax(min2,max2);
                            glm::vec3 O2 = (glm::vec3(max2) + glm::vec3(min2))/2.0f;
                            float R2 = glm::length(O2-glm::vec3(max2)) * object1->collisionRadiusСoefficient;

                            result = glm::length(O1-O2)<=(R1+R2);
                            if(result){
                                sphCollision(object1, object2, O1, O2, R1, R2, dt);
                                return result;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

void sphCollision(physicalObject* object1, physicalObject* object2, glm::vec3& O1, glm::vec3& O2, float& R1, float& R2, const float& dt)
{
    glm::vec3 v1 = object1->getVelocity();
    glm::vec3 v2 = object2->getVelocity();
    glm::vec3 n  = glm::normalize(O2-O1);
    float     e  = 1.0f;
    float     m1 = object1->getMass();
    float     m2 = object2->getMass();

    glm::vec3 un1 = ((1+e)*m2*glm::dot(v2,n) + (m1-e*m2)*glm::dot(v1,n))/(m1+m2) * n;
    glm::vec3 un2 = ((1+e)*m1*glm::dot(v1,n) + (m2-e*m1)*glm::dot(v2,n))/(m1+m2) * n;

    glm::vec3 ut1 = v1 - glm::dot(n,v1) * n;
    glm::vec3 ut2 = v2 - glm::dot(n,v2) * n;

    glm::vec3 u1 = un1 + ut1*(1-object1->getSurfaceFriction());
    glm::vec3 u2 = un2 + ut2*(1-object2->getSurfaceFriction());

    glm::vec3 w1 = -object1->getSurfaceFriction()*glm::cross(n,ut1)/R1;
    glm::vec3 w2 = -object2->getSurfaceFriction()*glm::cross(n,ut2)/R2;

    object1->setVelocity(u1);
    object2->setVelocity(u2);

    object1->setAngularVelocity(w1);
    object2->setAngularVelocity(w2);

//    object1->update(dt);
//    object2->update(dt);

//    findCollision(object1, object2, dt);
}

bool FindCollision1(physicalObject* object1, physicalObject* object2, const float& dt)
{
    bool result = false;
    for(auto *node1: object1->getModel()->linearNodes){
        if (node1->mesh) {
            glm::mat4x4 mat1 = object1->ModelMatrix() * node1->getMatrix();
            for(auto *primitive1: node1->mesh->primitives){
                glm::vec4 max1 = mat1 * glm::vec4(primitive1->bb.max,1.0f);
                glm::vec4 min1 = mat1 * glm::vec4(primitive1->bb.min,1.0f);
                glm::vec3 O1 = glm::vec3((max1.x+min1.x)/2.0f,(max1.y+min1.y)/2.0f,(max1.z+min1.z)/2.0f);
                float R1 = glm::length(O1-glm::vec3(max1));
                for(auto *node2: object2->getModel()->linearNodes){
                    if (node2->mesh) {
                        glm::mat4x4 mat2 = object2->ModelMatrix() * node2->getMatrix();
                        for(auto *primitive2: node2->mesh->primitives){
                            glm::vec4 max2 = mat2 * glm::vec4(primitive2->bb.max,1.0f);
                            glm::vec4 min2 = mat2 * glm::vec4(primitive2->bb.min,1.0f);
                            glm::vec3 O2 = glm::vec3((max2.x+min2.x)/2.0f,(max2.y+min2.y)/2.0f,(max2.z+min2.z)/2.0f);
                            float R2 = glm::length(O2-glm::vec3(max2));

                            result = glm::length(O1-O2)<=(R1+R2);

                            if(result){
                                glm::vec4 max1 = glm::vec4(primitive1->bb.max,1.0f);
                                glm::vec4 min1 = glm::vec4(primitive1->bb.min,1.0f);
                                glm::vec4 max2 = glm::vec4(primitive2->bb.max,1.0f);
                                glm::vec4 min2 = glm::vec4(primitive2->bb.min,1.0f);
                                result = FindCollision2(object1, min1, max1, mat1, O1, object2, min2, max2, mat2, O2, dt);
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

bool FindCollision2(physicalObject* object1, glm::vec4& min1, glm::vec4& max1, glm::mat4& mat1, glm::vec3& O1, physicalObject* object2, glm::vec4& min2, glm::vec4& max2, glm::mat4& mat2, glm::vec3& O2, const float& dt)
{
    bool result = false;
    float resT = 1.0f;
    glm::vec3 normal1 = glm::vec3(0.0f);
    glm::vec3 normal2 = glm::vec3(0.0f);
    glm::vec3 p1[8] = { glm::vec3(min1.x,min1.y,min1.z), glm::vec3(max1.x,min1.y,min1.z), glm::vec3(min1.x,max1.y,min1.z), glm::vec3(max1.x,max1.y,min1.z),
                        glm::vec3(min1.x,min1.y,max1.z), glm::vec3(max1.x,min1.y,max1.z), glm::vec3(min1.x,max1.y,max1.z), glm::vec3(max1.x,max1.y,max1.z) };
    glm::vec3 p2[8] = { glm::vec3(min2.x,min2.y,min2.z), glm::vec3(max2.x,min2.y,min2.z), glm::vec3(min2.x,max2.y,min2.z), glm::vec3(max2.x,max2.y,min2.z),
                        glm::vec3(min2.x,min2.y,max2.z), glm::vec3(max2.x,min2.y,max2.z), glm::vec3(min2.x,max2.y,max2.z), glm::vec3(max2.x,max2.y,max2.z) };
    glm::vec3 q1[8],q2[8];
    for(uint32_t i=0;i<8;i++){
        glm::vec4 P1 = mat1 * glm::vec4(p1[i],1.0f);
        glm::vec4 P2 = mat2 * glm::vec4(p2[i],1.0f);
        p1[i] = glm::vec3(P1.x,P1.y,P1.z);
        p2[i] = glm::vec3(P2.x,P2.y,P2.z);
        q1[i] = p1[i] - O1;
        q2[i] = p2[i] - O2;
    }

    glm::vec3 s1[6][4] = {  {p1[0],p1[1],p1[2],p1[3]},  {p1[0],p1[4],p1[2],p1[6]},  {p1[0],p1[1],p1[4],p1[5]},  {p1[7],p1[5],p1[3],p1[1]},  {p1[7],p1[3],p1[6],p1[2]},  {p1[7],p1[6],p1[5],p1[4]}};
    glm::vec3 s2[6][4] = {  {p2[0],p2[1],p2[2],p2[3]},  {p2[0],p2[4],p2[2],p2[6]},  {p2[0],p2[1],p2[4],p2[5]},  {p2[7],p2[5],p2[3],p2[1]},  {p2[7],p2[3],p2[6],p2[2]},  {p2[7],p2[6],p2[5],p2[4]}};
    glm::vec3 Max1[6], Min1[6], n1[6];
    for(uint32_t k=0;k<6;k++){
        Max1[k] = findMax(s1[k][0],s1[k][1],s1[k][2],s1[k][3]);
        Min1[k] = findMin(s1[k][0],s1[k][1],s1[k][2],s1[k][3]);
        glm::mat3 m = glm::inverse(glm::mat3(s1[k][0],s1[k][1],s1[k][2]));
        n1[k] = - glm::vec3(m[0][0]+m[0][1]+m[0][2],m[1][0]+m[1][1]+m[1][2],m[2][0]+m[2][1]+m[2][2]);
    }
    for(uint32_t i=0;i<8;i++){
        for(uint32_t k=0;k<6;k++){
            float t = -(1.0f + glm::dot(n1[k],O2))/glm::dot(n1[k],q2[i]);
            glm::vec3 point = O2 + q2[i]*t;
            if(t<1.0f&&t>0.0f){
                bool xcond = (point.x>=Min1[k].x&&point.x<=Max1[k].x);
                bool ycond = (point.y>=Min1[k].y&&point.y<=Max1[k].y);
                bool zcond = (point.z>=Min1[k].z&&point.z<=Max1[k].z);
                if(xcond&&ycond&&zcond){
                    resT = t;
                    normal1 = glm::normalize(n1[k]);
                    normal2 = glm::normalize(n1[k]);
                    result = true;
                    break;
                }
            }
        }
    }
    if(!result){
        normal1 = glm::vec3(0.0f);
        normal2 = glm::vec3(0.0f);
        glm::vec3 Max2[6], Min2[6], n2[6];
        for(uint32_t k=0;k<6;k++){
            Max2[k] = findMax(s2[k][0],s2[k][1],s2[k][2],s2[k][3]);
            Min2[k] = findMin(s2[k][0],s2[k][1],s2[k][2],s2[k][3]);
            glm::mat3 m = glm::inverse(glm::mat3(s2[k][0],s2[k][1],s2[k][2]));
            n2[k] = - glm::vec3(m[0][0]+m[0][1]+m[0][2],m[1][0]+m[1][1]+m[1][2],m[2][0]+m[2][1]+m[2][2]);
        }
        for(uint32_t i=0;i<8;i++){
            for(uint32_t k=0;k<6;k++){
                float t = -(1.0f + glm::dot(n2[k],O1))/glm::dot(n2[k],q1[i]);
                glm::vec3 point = O1 + q1[i]*t;
                if(t<1.0f&&t>0.0f){
                    bool xcond = (point.x>=Min2[k].x&&point.x<=Max2[k].x);
                    bool ycond = (point.y>=Min2[k].y&&point.y<=Max2[k].y);
                    bool zcond = (point.z>=Min2[k].z&&point.z<=Max2[k].z);
                    if(xcond&&ycond&&zcond){
                        resT = t;
                        normal1 = glm::normalize(n2[k]);
                        normal2 = glm::normalize(n2[k]);
                        result = true;
                        break;
                    }
                }
            }
        }
    }
    if(result){
        glm::vec3 v1 = object1->getVelocity();
        glm::vec3 v2 = object2->getVelocity();
        glm::vec3 n1  = - glm::normalize(normal1);
        glm::vec3 n2  = - glm::normalize(normal1);
        float     e  = 1.0f;
        float     m1 = object1->getMass();
        float     m2 = object2->getMass();
        glm::vec3 un1 = ((1+e)*m2*glm::dot(v2,n2) + (m1-e*m2)*glm::dot(v1,n1))/(m1+m2) * n1;
        glm::vec3 un2 = ((1+e)*m1*glm::dot(v1,n1) + (m2-e*m1)*glm::dot(v2,n2))/(m1+m2) * n2;
        glm::vec3 ut1 = v1 - glm::dot(n1,v1) * n1;
        glm::vec3 ut2 = v2 - glm::dot(n2,v2) * n2;
        glm::vec3 u1 = un1 + ut1;
        glm::vec3 u2 = un2 + ut2;
        object1->setVelocity(u1);
        object2->setVelocity(u2);
        if(resT<0.8f){
            glm::vec3 ds1 = 0.1f * object1->getVelocity();
            object1->translate(ds1);
            glm::vec3 ds2 = 0.1f * object2->getVelocity();
            object2->translate(ds2);
            while(findCollision1(object1,object2,dt)){
                    glm::vec3 ds1 = 0.1f * m2/(m1+m2) * object1->getVelocity();
                    object1->translate(ds1);
                    glm::vec3 ds2 = 0.1f * m1/(m1+m2) * object2->getVelocity();
                    object2->translate(ds2);
            }
        }
    }
    return result;
}

bool findCollision1(physicalObject* object1, physicalObject* object2, const float& dt)
{
    bool result = false;
    for(auto *node1: object1->getModel()->linearNodes){
        if (node1->mesh) {
            glm::mat4x4 mat1 = object1->ModelMatrix() * node1->getMatrix();
            for(auto *primitive1: node1->mesh->primitives){
                glm::vec4 max1 = mat1 * glm::vec4(primitive1->bb.max,1.0f);
                glm::vec4 min1 = mat1 * glm::vec4(primitive1->bb.min,1.0f);
                glm::vec3 O1 = (glm::vec3(max1) + glm::vec3(min1))/2.0f;
                float R1 = glm::length(O1-glm::vec3(max1));
                for(auto *node2: object2->getModel()->linearNodes){
                    if (node2->mesh) {
                        glm::mat4x4 mat2 = object2->ModelMatrix() * node2->getMatrix();
                        for(auto *primitive2: node2->mesh->primitives){
                            glm::vec4 max2 = mat2 * glm::vec4(primitive2->bb.max,1.0f);
                            glm::vec4 min2 = mat2 * glm::vec4(primitive2->bb.min,1.0f);
                            glm::vec3 O2 = (glm::vec3(max2) + glm::vec3(min2))/2.0f;
                            float R2 = glm::length(O2-glm::vec3(max2));
                            result = glm::length(O1-O2)<=(R1+R2);
                            if(result){
                                glm::vec4 max1 = glm::vec4(primitive1->bb.max,1.0f);
                                glm::vec4 min1 = glm::vec4(primitive1->bb.min,1.0f);
                                glm::vec4 max2 = glm::vec4(primitive2->bb.max,1.0f);
                                glm::vec4 min2 = glm::vec4(primitive2->bb.min,1.0f);
                                result = findCollision2(object1, min1, max1, mat1, O1, object2, min2, max2, mat2, O2, dt);
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

bool findCollision2(physicalObject* object1, glm::vec4& min1, glm::vec4& max1, glm::mat4& mat1, glm::vec3& O1, physicalObject* object2, glm::vec4& min2, glm::vec4& max2, glm::mat4& mat2, glm::vec3& O2, const float& dt)
{
    bool result = false;
    glm::vec3 p1[8] = { glm::vec3(min1.x,min1.y,min1.z), glm::vec3(max1.x,min1.y,min1.z), glm::vec3(min1.x,max1.y,min1.z), glm::vec3(max1.x,max1.y,min1.z),
                        glm::vec3(min1.x,min1.y,max1.z), glm::vec3(max1.x,min1.y,max1.z), glm::vec3(min1.x,max1.y,max1.z), glm::vec3(max1.x,max1.y,max1.z) };
    glm::vec3 p2[8] = { glm::vec3(min2.x,min2.y,min2.z), glm::vec3(max2.x,min2.y,min2.z), glm::vec3(min2.x,max2.y,min2.z), glm::vec3(max2.x,max2.y,min2.z),
                        glm::vec3(min2.x,min2.y,max2.z), glm::vec3(max2.x,min2.y,max2.z), glm::vec3(min2.x,max2.y,max2.z), glm::vec3(max2.x,max2.y,max2.z) };
    for(uint32_t i=0;i<8;i++){
        glm::vec4 P1 = mat1 * glm::vec4(p1[i],1.0f);
        glm::vec4 P2 = mat2 * glm::vec4(p2[i],1.0f);
        p1[i] = glm::vec3(P1.x,P1.y,P1.z);
        p2[i] = glm::vec3(P2.x,P2.y,P2.z);
    }
    glm::vec3 q1[8] = { p1[0] - O1, p1[1] - O1, p1[2] - O1, p1[3] - O1, p1[4] - O1, p1[5] - O1, p1[6] - O1, p1[7] - O1};
    glm::vec3 q2[8] = { p2[0] - O2, p2[1] - O2, p2[2] - O2, p2[3] - O2, p2[4] - O2, p2[5] - O2, p2[6] - O2, p2[7] - O2};
    glm::vec3 s1[6][4] = {  {p1[0],p1[1],p1[2],p1[3]},  {p1[0],p1[4],p1[2],p1[6]},  {p1[0],p1[1],p1[4],p1[5]},  {p1[7],p1[5],p1[3],p1[1]},  {p1[7],p1[3],p1[6],p1[2]},  {p1[7],p1[6],p1[5],p1[4]}};
    glm::vec3 s2[6][4] = {  {p2[0],p2[1],p2[2],p2[3]},  {p2[0],p2[4],p2[2],p2[6]},  {p2[0],p2[1],p2[4],p2[5]},  {p2[7],p2[5],p2[3],p2[1]},  {p2[7],p2[3],p2[6],p2[2]},  {p2[7],p2[6],p2[5],p2[4]}};
    glm::vec3 Max1[6], Min1[6], n1[6];
    for(uint32_t k=0;k<6;k++){
        Max1[k] = findMax(s1[k][0],s1[k][1],s1[k][2],s1[k][3]);
        Min1[k] = findMin(s1[k][0],s1[k][1],s1[k][2],s1[k][3]);
        glm::mat3 m = glm::inverse(glm::mat3(s1[k][0],s1[k][1],s1[k][2]));
        n1[k] =  - glm::vec3(m[0][0]+m[0][1]+m[0][2],m[1][0]+m[1][1]+m[1][2],m[2][0]+m[2][1]+m[2][2]);
    }
    for(uint32_t i=0;i<8;i++){
        for(uint32_t k=0;k<6;k++){
            float t = -(1.0f + glm::dot(n1[k],O2))/glm::dot(n1[k],q2[i]);
            glm::vec3 point = O2 + q2[i]*t;
            if(t<1.0f&&t>0.0f){
                bool xcond = (point.x>=Min1[k].x&&point.x<=Max1[k].x);
                bool ycond = (point.y>=Min1[k].y&&point.y<=Max1[k].y);
                bool zcond = (point.z>=Min1[k].z&&point.z<=Max1[k].z);
                if(xcond&&ycond&&zcond){
                    result = true;
                    break;
                }
            }
        }
    }
    if(!result){
        glm::vec3 Max2[6], Min2[6], n2[6];
        for(uint32_t k=0;k<6;k++){
            Max2[k] = findMax(s2[k][0],s2[k][1],s2[k][2],s2[k][3]);
            Min2[k] = findMin(s2[k][0],s2[k][1],s2[k][2],s2[k][3]);
            glm::mat3 m = glm::inverse(glm::mat3(s2[k][0],s2[k][1],s2[k][2]));
            n2[k] = - glm::vec3(m[0][0]+m[0][1]+m[0][2],m[1][0]+m[1][1]+m[1][2],m[2][0]+m[2][1]+m[2][2]);
        }
        for(uint32_t i=0;i<8;i++){
            for(uint32_t k=0;k<6;k++){
                float t = -(1.0f + glm::dot(n2[k],O1))/glm::dot(n2[k],q1[i]);
                glm::vec3 point = O1 + q1[i]*t;
                if(t<1.0f&&t>0.0f){
                    bool xcond = (point.x>=Min2[k].x&&point.x<=Max2[k].x);
                    bool ycond = (point.y>=Min2[k].y&&point.y<=Max2[k].y);
                    bool zcond = (point.z>=Min2[k].z&&point.z<=Max2[k].z);
                    if(xcond&&ycond&&zcond){
                        result = true;
                        break;
                    }
                }
            }
        }
    }
    return result;
}

glm::vec3 findMin(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3)
{
    glm::vec3 min = v0;
    if(v1.x<min.x)  min.x = v1.x;
    if(v2.x<min.x)  min.x = v2.x;
    if(v3.x<min.x)  min.x = v3.x;
    if(v1.y<min.y)  min.y = v1.y;
    if(v2.y<min.y)  min.y = v2.y;
    if(v3.y<min.y)  min.y = v3.y;
    if(v1.z<min.z)  min.z = v1.z;
    if(v2.z<min.z)  min.z = v2.z;
    if(v3.z<min.z)  min.z = v3.z;
    return min;
}

glm::vec3 findMax(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3)
{
    glm::vec3 max = v0;
    if(v1.x>max.x)  max.x = v1.x;
    if(v2.x>max.x)  max.x = v2.x;
    if(v3.x>max.x)  max.x = v3.x;
    if(v1.y>max.y)  max.y = v1.y;
    if(v2.y>max.y)  max.y = v2.y;
    if(v3.y>max.y)  max.y = v3.y;
    if(v1.z>max.z)  max.z = v1.z;
    if(v2.z>max.z)  max.z = v2.z;
    if(v3.z>max.z)  max.z = v3.z;
    return max;
}

bool FindCollision1(object* object1, physicalObject* object2, const float& dt)
{
    bool result = false;
    for(auto *node1: object1->getModel()->linearNodes){
        if (node1->mesh) {
            glm::mat4x4 mat1 = object1->ModelMatrix() * node1->getMatrix();
            for(auto *primitive1: node1->mesh->primitives){
                glm::vec4 max1 = mat1 * glm::vec4(primitive1->bb.max,1.0f);
                glm::vec4 min1 = mat1 * glm::vec4(primitive1->bb.min,1.0f);
                glm::vec3 O1 = (glm::vec3(max1) + glm::vec3(min1))/2.0f;
                float R1 = glm::length(O1-glm::vec3(max1));
                for(auto *node2: object2->getModel()->linearNodes){
                    if (node2->mesh) {
                        glm::mat4x4 mat2 = object2->ModelMatrix() * node2->getMatrix();
                        for(auto *primitive2: node2->mesh->primitives){
                            glm::vec4 max2 = mat2 * glm::vec4(primitive2->bb.max,1.0f);
                            glm::vec4 min2 = mat2 * glm::vec4(primitive2->bb.min,1.0f);
                            glm::vec3 O2 = (glm::vec3(max2) + glm::vec3(min2))/2.0f;
                            float R2 = glm::length(O2-glm::vec3(max2));
                            result = glm::length(O1-O2)<=(R1+R2);
                            if(result){
                                result = true;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

bool FindCollision2(object* object1, glm::vec4& min1, glm::vec4& max1, glm::mat4& mat1, glm::vec3& O1, physicalObject* object2, glm::vec4& min2, glm::vec4& max2, glm::mat4& mat2, glm::vec3& O2, const float& dt)
{
    bool result = false;
    glm::vec3 p1[8] = { glm::vec3(min1.x,min1.y,min1.z), glm::vec3(max1.x,min1.y,min1.z), glm::vec3(min1.x,max1.y,min1.z), glm::vec3(max1.x,max1.y,min1.z),
                        glm::vec3(min1.x,min1.y,max1.z), glm::vec3(max1.x,min1.y,max1.z), glm::vec3(min1.x,max1.y,max1.z), glm::vec3(max1.x,max1.y,max1.z) };
    glm::vec3 p2[8] = { glm::vec3(min2.x,min2.y,min2.z), glm::vec3(max2.x,min2.y,min2.z), glm::vec3(min2.x,max2.y,min2.z), glm::vec3(max2.x,max2.y,min2.z),
                        glm::vec3(min2.x,min2.y,max2.z), glm::vec3(max2.x,min2.y,max2.z), glm::vec3(min2.x,max2.y,max2.z), glm::vec3(max2.x,max2.y,max2.z) };
    for(uint32_t i=0;i<8;i++){
        glm::vec4 P1 = mat1 * glm::vec4(p1[i],1.0f);
        glm::vec4 P2 = mat2 * glm::vec4(p2[i],1.0f);
        p1[i] = glm::vec3(P1.x,P1.y,P1.z);
        p2[i] = glm::vec3(P2.x,P2.y,P2.z);
    }
    glm::vec3 q1[8] = { p1[0] - O1, p1[1] - O1, p1[2] - O1, p1[3] - O1, p1[4] - O1, p1[5] - O1, p1[6] - O1, p1[7] - O1};
    glm::vec3 q2[8] = { p2[0] - O2, p2[1] - O2, p2[2] - O2, p2[3] - O2, p2[4] - O2, p2[5] - O2, p2[6] - O2, p2[7] - O2};
    glm::vec3 s1[6][4] = {  {p1[0],p1[1],p1[2],p1[3]},  {p1[0],p1[4],p1[2],p1[6]},  {p1[0],p1[1],p1[4],p1[5]},  {p1[7],p1[5],p1[3],p1[1]},  {p1[7],p1[3],p1[6],p1[2]},  {p1[7],p1[6],p1[5],p1[4]}};
    glm::vec3 s2[6][4] = {  {p2[0],p2[1],p2[2],p2[3]},  {p2[0],p2[4],p2[2],p2[6]},  {p2[0],p2[1],p2[4],p2[5]},  {p2[7],p2[5],p2[3],p2[1]},  {p2[7],p2[3],p2[6],p2[2]},  {p2[7],p2[6],p2[5],p2[4]}};
    glm::vec3 Max1[6], Min1[6], n1[6];
    for(uint32_t k=0;k<6;k++){
        Max1[k] = findMax(s1[k][0],s1[k][1],s1[k][2],s1[k][3]);
        Min1[k] = findMin(s1[k][0],s1[k][1],s1[k][2],s1[k][3]);
        glm::mat3 m = glm::inverse(glm::mat3(s1[k][0],s1[k][1],s1[k][2]));
        n1[k] =  - glm::vec3(m[0][0]+m[0][1]+m[0][2],m[1][0]+m[1][1]+m[1][2],m[2][0]+m[2][1]+m[2][2]);
    }
    for(uint32_t i=0;i<8;i++){
        for(uint32_t k=0;k<6;k++){
            float t = -(1.0f + glm::dot(n1[k],O2))/glm::dot(n1[k],q2[i]);
            glm::vec3 point = O2 + q2[i]*t;
            if(t<1.0f&&t>0.0f){
                bool xcond = (point.x>=Min1[k].x&&point.x<=Max1[k].x);
                bool ycond = (point.y>=Min1[k].y&&point.y<=Max1[k].y);
                bool zcond = (point.z>=Min1[k].z&&point.z<=Max1[k].z);
                if(xcond&&ycond&&zcond){
                    result = true;
                    break;
                }
            }
        }
    }
    return result;
}