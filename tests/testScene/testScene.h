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
namespace moon::transformational {
class SpotLight;
class IsotropicLight;
class BaseObject;
class Group;
class BaseCamera;
class SkyboxObject;
}
namespace moon::utils {
class Cursor;
}

class testScene : public scene
{
private:
    bool& framebufferResized;

    std::filesystem::path ExternalPath;
    moon::math::Vector<uint32_t,2> extent{0};
    moon::math::Vector<double,2> mousePos{0.0};
    float globalTime{0.0f};
    bool enableScatteringRefraction{true};
    int ufoCounter{0};

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
    std::shared_ptr<moon::utils::Cursor> cursor{ nullptr };

    uint32_t resourceCount{0};
    uint32_t imageCount{0};

    std::unordered_map<std::string, std::shared_ptr<moon::transformational::BaseCamera>> cameras;
    std::unordered_map<std::string, std::shared_ptr<moon::deferredGraphics::DeferredGraphics>> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;
#endif

    std::unordered_map<std::string, std::shared_ptr<moon::interfaces::Model>> models;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::BaseObject>> objects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::BaseObject>> staticObjects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::SkyboxObject>> skyboxObjects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Group>> groups;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::IsotropicLight>> lightPoints;
    std::vector<std::shared_ptr<moon::transformational::SpotLight>> lightSources;
    moon::transformational::BaseObject* controledObject{nullptr};

    bool            controledObjectEnableOutlighting{true};
    float           controledObjectOutlightingColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::string     controledObjectName{"none"};

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);

    void create();
    void loadModels();
    void createLight();
    void createObjects();
public:
    testScene(moon::graphicsManager::GraphicsManager *app, GLFWwindow* window, uint32_t width, uint32_t height, const std::filesystem::path& ExternalPath, bool& framebufferResized);

    void resize(uint32_t width, uint32_t height) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTSCENE_H
