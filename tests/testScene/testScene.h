#ifndef TESTSCENE_H
#define TESTSCENE_H

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <glfw3.h>

#include "scene.h"
#include "vector.h"

#include "controller.h"

class deferredGraphics;
class imguiGraphics;
class model;
class graphicsManager;
class spotLight;
class isotropicLight;
class baseObject;
class group;
class baseCamera;
class skyboxObject;
class plyModel;
class transformational;

//#define SECOND_VIEW_WINDOW
//#define IMGUI_GRAPHICS

class testScene : public scene
{
private:
    std::filesystem::path   ExternalPath;
    vector<uint32_t,2>      extent{0};
    vector<double,2>        mousePos{0.0};
    float                   globalTime{0.0f};
    float                   timeScale{1.0f};
    float                   minAmbientFactor{0.05f};
    bool                    enableScatteringRefraction{true};
    int                     ufoCounter{0};

    transformational*                                   controledObject{nullptr};
    GLFWwindow*                                         window{nullptr};
    graphicsManager*                                    app{nullptr};
    controller*                                         mouse{nullptr};
    controller*                                         board{nullptr};

    std::unordered_map<std::string, baseCamera*>        cameras;
    std::unordered_map<std::string, deferredGraphics*>  graphics;
    imguiGraphics*                                      gui;

    std::unordered_map<std::string, model*>             models;
    std::unordered_map<std::string, baseObject*>        objects;
    std::unordered_map<std::string, baseObject*>        staticObjects;
    std::unordered_map<std::string, skyboxObject*>      skyboxObjects;
    std::unordered_map<std::string, group*>             groups;
    std::unordered_map<std::string, isotropicLight*>    lightPoints;
    std::vector<spotLight*>                             lightSources;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    testScene(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath);
    ~testScene();

    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
    void destroy() override;
};

#endif // TESTSCENE_H
