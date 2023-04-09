#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

class deferredGraphics;
class model;
class graphicsManager;
class spotLight;
class isotropicLight;
class object;
class group;
class camera;
class skyboxObject;
class GLFWwindow;

void scrol(GLFWwindow* window, double xoffset, double yoffset);

class scene
{
private:
    uint32_t    WIDTH;
    uint32_t    HEIGHT;

    float       globalTime = 0.0f;
    float       timeScale = 1.0f;
    float       minAmbientFactor = 0.05f;

    double      xMpos, yMpos;
    double      angx=0.0, angy=0.0;
    bool        mouse1Stage = 0;
    bool        backRStage = 0;
    bool        backTStage = 0;
    bool        backYStage = 0;
    bool        backNStage = 0;
    bool        backBStage = 0;
    bool        backOStage = 0;
    bool        backIStage = 0;
    bool        backGStage = 0;
    bool        backHStage = 0;

    uint32_t    controledGroup = 0;
    uint32_t    lightPointer = 10;

    camera*                                         cameras;
    skyboxObject*                                   skyboxObject1;
    skyboxObject*                                   skyboxObject2;

    std::vector<model                   *>          gltfModel;
    std::vector<object                  *>          object3D;
    std::vector<spotLight               *>          lightSources;
    std::vector<isotropicLight          *>          lightPoints;
    std::vector<group                   *>          groups;

    graphicsManager*    app;
    deferredGraphics*   graphics;
    GLFWwindow*         window;

    std::string ExternalPath;
    std::string ZERO_TEXTURE;
    std::string ZERO_TEXTURE_WHITE;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    scene(graphicsManager *app, deferredGraphics* graphics, GLFWwindow* window, std::string ExternalPath);
    void createScene(uint32_t WIDTH, uint32_t HEIGHT, camera* cameraObject);
    void updateFrame(uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT);
    void destroyScene();

    bool framebufferResized = false;
};

#endif // SCENE_H
