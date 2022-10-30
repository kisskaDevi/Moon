#ifndef SCENE2_H
#define SCENE2_H

#include "core/graphicsManager.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/transformational/object.h"
#include "core/transformational/group.h"
#include "core/transformational/camera.h"
#include "libs/stb-master/stb_image.h"

class deferredGraphicsInterface;

class scene2
{
private:
    std::string ExternalPath;
    uint32_t    WIDTH;
    uint32_t    HEIGHT;

    float       globalTime = 0.0f;
    float       timeScale = 1.0f;
    float       minAmbientFactor = 0.05f;

    double      xMpos, yMpos;
    double      angx=0.0, angy=0.0;

    bool backStage[2] = {0};
    bool backStageSpace = 0;

    bool cameraAnimation = false;
    float cameraTimer = 0.0f;
    float cameraAnimationTime = 2.0f;
    uint32_t cameraPoint = 2.0f;

    dualQuaternion<float>                           dQuat[2];
    quaternion<float>                               quatX[2];
    quaternion<float>                               quatY[2];

    std::string ZERO_TEXTURE;
    std::string ZERO_TEXTURE_WHITE;

    camera*                                         cameras;
    object*                                         skyboxObject;

    std::vector<std::vector<gltfModel   *>>         gltfModel;
    std::vector<object                  *>          object3D;
    std::vector<spotLight               *>          lightSource;
    std::vector<pointLight              *>          lightPoint;
    std::vector<group                   *>          groups;

    graphicsManager*            app;
    deferredGraphicsInterface*  graphics;

    void mouseEvent(GLFWwindow* window, float frameTime);
    void keyboardEvent(GLFWwindow* window, float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    scene2(graphicsManager *app, deferredGraphicsInterface* graphics, std::string ExternalPath);
    void createScene(uint32_t WIDTH, uint32_t HEIGHT);
    void updateFrame(GLFWwindow* window, uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT);
    void destroyScene();
};
#endif // SCENE2_H
