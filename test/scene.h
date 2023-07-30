#ifndef SCENE_H
#define SCENE_H

#include <filesystem>
#include <vector>

class deferredGraphics;
class model;
class graphicsManager;
class spotLight;
class isotropicLight;
class baseObject;
class group;
class baseCamera;
class skyboxObject;
struct GLFWwindow;
class plyModel;
class transformational;

void scrol(GLFWwindow* window, double xoffset, double yoffset);

class scene
{
private:
    uint32_t    WIDTH{0}, HEIGHT{0};

    float       globalTime = 0.0f;
    float       timeScale = 1.0f;
    float       minAmbientFactor = 0.05f;

    double      xMpos{0.0}, yMpos{0.0};
    bool        mouse1Stage = 0;
    bool        backTStage = 0;
    bool        backYStage = 0;
    bool        backNStage = 0;
    bool        backBStage = 0;
    bool        backOStage = 0;
    bool        backIStage = 0;
    bool        backGStage = 0;
    bool        backHStage = 0;

    transformational*   controledObject{nullptr};

    baseCamera*         cameras{nullptr};

    std::vector<model                   *>          models;
    std::vector<baseObject              *>          object3D;
    std::vector<baseObject              *>          staticObject3D;
    std::vector<skyboxObject            *>          skyboxObjects;
    std::vector<spotLight               *>          lightSources;
    std::vector<isotropicLight          *>          lightPoints;
    std::vector<group                   *>          groups;

    GLFWwindow*                         window;
    graphicsManager*                    app;
    std::vector<deferredGraphics*>      graphics;

    std::filesystem::path ExternalPath;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    scene(graphicsManager *app, std::vector<deferredGraphics*> graphics, GLFWwindow* window, const std::filesystem::path& ExternalPath);
    void createScene(uint32_t WIDTH, uint32_t HEIGHT, baseCamera* cameraObject);
    void updateFrame(uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT);
    void destroyScene();
};

#endif // SCENE_H
