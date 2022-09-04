#ifndef SCENE_H
#define SCENE_H

#include "core/vulkanCore.h"
#include "core/texture.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/transformational/object.h"
#include "core/transformational/group.h"
#include "core/transformational/camera.h"
#include "libs/stb-master/stb_image.h"

class scene
{
private:
    uint32_t WIDTH;
    uint32_t HEIGHT;

    bool  animate = true;
    float globalTime = 0.0f;

    float spotAngle = 90.0f;
    bool updateLightCone = false;

    float minAmbientFactor = 0.05f;

    uint32_t controledGroup = 0;

    double   xMpos, yMpos;
    double   angx=0.0, angy=0.0;
    bool     mouse1Stage = 0;
    bool     backRStage = 0;
    bool     backTStage = 0;
    bool     backYStage = 0;
    bool     backNStage = 0;
    bool     backBStage = 0;
    bool     backOStage = 0;
    bool     backIStage = 0;
    bool     backGStage = 0;
    bool     backHStage = 0;

    uint32_t lightPointer = 10;

    camera *cameras;
    bool updateCamera = false;
    float cameraAngle = 45.0f;

    object *skyboxObject;

    std::vector<std::vector<gltfModel   *>>         gltfModel;
    std::vector<object                  *>          object3D;
    std::vector<light<spotLight>        *>          lightSource;
    std::vector<light<pointLight>       *>          lightPoint;
    std::vector<group                   *>          groups;

    void mouseEvent(VkApplication* app, GLFWwindow* window, float frameTime);
    void keyboardEvent(VkApplication* app, GLFWwindow* window, float frameTime);
    void updates(VkApplication* app, float frameTime);

    void loadModels(VkApplication* app);
    void createLight(VkApplication* app);
    void createObjects(VkApplication* app);
    void scrol(GLFWwindow* window, double xoffset, double yoffset);
public:
    scene();
    void createScene(VkApplication *app, uint32_t WIDTH, uint32_t HEIGHT);
    void updateFrame(VkApplication *app, GLFWwindow* window, uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT);
    void destroyScene(VkApplication *app);
};

#endif // SCENE_H
