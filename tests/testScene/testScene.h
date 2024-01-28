#ifndef TESTSCENE_H
#define TESTSCENE_H

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
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
#define IMGUI_GRAPHICS

class testScene : public scene
{
private:
    bool& framebufferResized;

    std::filesystem::path   ExternalPath;
    vector<uint32_t,2>      extent{0};
    vector<double,2>        mousePos{0.0};
    float                   globalTime{0.0f};
    bool                    enableScatteringRefraction{true};
    int                     ufoCounter{0};

    float                   blitFactor = 1.5f;
    float                   farBlurDepth = 1.0f;
    float                   minAmbientFactor{0.05f};
    float                   animationSpeed{1.0f};

    GLFWwindow*                                         window{nullptr};
    graphicsManager*                                    app{nullptr};
    std::shared_ptr<controller>                         mouse{nullptr};
    std::shared_ptr<controller>                         board{nullptr};

    std::unordered_map<std::string, std::shared_ptr<baseCamera>>        cameras;
    std::unordered_map<std::string, std::shared_ptr<deferredGraphics>>  graphics;
    std::shared_ptr<imguiGraphics>                                      gui;

    std::unordered_map<std::string, std::shared_ptr<model>>             models;
    std::unordered_map<std::string, std::shared_ptr<baseObject>>        objects;
    std::unordered_map<std::string, std::shared_ptr<baseObject>>        staticObjects;
    std::unordered_map<std::string, std::shared_ptr<skyboxObject>>      skyboxObjects;
    std::unordered_map<std::string, std::shared_ptr<group>>             groups;
    std::unordered_map<std::string, std::shared_ptr<isotropicLight>>    lightPoints;
    std::vector<std::shared_ptr<spotLight>>                             lightSources;

    bool            controledObjectEnableOutlighting{true};
    float           controledObjectOutlightingColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::string     controledObjectName{"none"};
    baseObject*     controledObject{nullptr};

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    testScene(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath, bool& framebufferResized);

    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTSCENE_H
