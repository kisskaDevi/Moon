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

// #define SECOND_VIEW_WINDOW
#define IMGUI_GRAPHICS

namespace moon::interfaces { class Model;}
namespace moon::graphicsManager { class GraphicsManager;}
namespace moon::imguiGraphics { class ImguiGraphics;}
namespace moon::deferredGraphics { class DeferredGraphics;}

class spotLight;
class isotropicLight;
class baseObject;
class group;
class baseCamera;
class skyboxObject;
class transformational;

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
    std::string             screenshot;
    uint32_t                primitiveNumber = std::numeric_limits<uint32_t>::max();

#ifdef SECOND_VIEW_WINDOW
    vector<float,2>      viewOffset{0.5f,0.5f};
    vector<float,2>      viewExtent{0.33f,0.33f};
#endif

    GLFWwindow* window{nullptr};
    moon::graphicsManager::GraphicsManager* app{nullptr};
    std::shared_ptr<controller> mouse{nullptr};
    std::shared_ptr<controller> board{nullptr};

    uint32_t resourceCount{0};
    uint32_t imageCount{0};

    std::unordered_map<std::string, std::shared_ptr<baseCamera>> cameras;
    std::unordered_map<std::string, std::shared_ptr<moon::deferredGraphics::DeferredGraphics>> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;
#endif

    std::unordered_map<std::string, std::shared_ptr<moon::interfaces::Model>> models;
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
    testScene(moon::graphicsManager::GraphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath, bool& framebufferResized);

    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTSCENE_H
