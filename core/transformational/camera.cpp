#include "camera.h"

camera::camera()
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    viewMatrix = glm::mat4x4(1.0f);
    projMatrix = glm::mat4x4(1.0f);

    quat = quaternion<float>(1.0f,0.0f,0.0f,0.0f);
    quatX = quaternion<float>(1.0f,0.0f,0.0f,0.0f);
    quatY = quaternion<float>(1.0f,0.0f,0.0f,0.0f);
}

camera::~camera()
{

}


void camera::setProjMatrix(const glm::mat4 & proj)
{
    projMatrix = proj;
}

glm::mat4x4 camera::getProjMatrix() const
{
    return projMatrix;
}

void camera::setPosition(const glm::vec3 & translate)
{
    m_translate = translate;
    updateViewMatrix();
}

void camera::setRotation(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax));
    updateViewMatrix();
}

void camera::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void camera::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void camera::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    quat = convert(ang,ax);
    updateViewMatrix();
}

void camera::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void camera::updateViewMatrix()
{
    //if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
    //{
    //    rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    //}
    //glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),-m_translate);

    glm::mat3x3 R = convert(quat);

    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    rotateMatrix[0][0] = R[0][0];   rotateMatrix[0][1] = R[0][1];   rotateMatrix[0][2] = R[0][2];
    rotateMatrix[1][0] = R[1][0];   rotateMatrix[1][1] = R[1][1];   rotateMatrix[1][2] = R[1][2];
    rotateMatrix[2][0] = R[2][0];   rotateMatrix[2][1] = R[2][1];   rotateMatrix[2][2] = R[2][2];
    rotateMatrix[3][3] = 1.0f;

    dQuat = convert(quat,quaternion<float>(0.0f,m_translate.x,m_translate.y,m_translate.z));
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);

    viewMatrix = glm::inverse(m_globalTransform) * glm::inverse(transformMatrix);
}

glm::mat4x4 camera::getViewMatrix() const
{
    return viewMatrix;
}

glm::vec3 camera::getTranslate() const
{
    return m_translate;
}

void camera::rotateX(const float & ang ,const glm::vec3 & ax)
{
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;

    quatX = convert(ang,ax) * quatX;
    quat = quatX*quatY;

    updateViewMatrix();
}

void camera::rotateY(const float & ang ,const glm::vec3 & ax)
{
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    m_rotate = m_rotateX * m_rotateY;

    quatY = convert(ang,ax) * quatY;
    quat = quatX*quatY;

    updateViewMatrix();
}



void                    camera::setDualQuaternion(const dualQuaternion<float>& dQuat)
{
    this->dQuat = dQuat;
    glm::mat4x4 transformMatrix = convert(this->dQuat);
    this->quat = dQuat.rotation();
    m_translate = dQuat.translation().vector();

    viewMatrix = glm::inverse(m_globalTransform) * glm::inverse(transformMatrix);
}

dualQuaternion<float>   camera::getDualQuaternion()const
{
    return dQuat;
}

void                    camera::setQuaternion(const quaternion<float>& quat)
{
    this->quat = quat;

    updateViewMatrix();
}

void                    camera::setQuaternions(const quaternion<float>& quatX, const quaternion<float>& quatY)
{
    this->quatX = quatX;
    this->quatY = quatY;
}

quaternion<float>       camera::getquatX()const
{
    return quatX;
}

quaternion<float>       camera::getquatY()const
{
    return quatY;
}
