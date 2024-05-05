#ifndef TESTPOS_H
#define TESTPOS_H

#include <glfw3.h>

#include <filesystem>
#include <memory>

#include "scene.h"
#include "controller.h"
#include "vector.h"

#include "transformational/camera.h"
#include "models/model.h"

#define IMGUI_GRAPHICS

namespace moon::graphicsManager { class GraphicsManager;}
namespace moon::imguiGraphics { class ImguiGraphics;}
namespace moon::rayTracingGraphics { class RayTracingGraphics;}

class testCuda : public scene
{
private:
    bool& framebufferResized;

    std::filesystem::path   ExternalPath;
    vector<uint32_t,2>      extent{0};
    vector<double,2>        mousePos{0.0};

    float                   focus = 0.049f;
    float                   blitFactor = 1.5f;
    std::string             screenshot;

    moon::graphicsManager::GraphicsManager *app{nullptr};
    GLFWwindow* window{nullptr};
    cuda::devicep<cuda::camera> cam;
    cuda::camera hostcam = cuda::camera(cuda::ray(cuda::vec4f(2.0f, 0.0f, 2.0f, 1.0f), cuda::vec4f(-1.0f, 0.0f, -1.0f, 0.0f)), 1.0f);

    std::shared_ptr<controller> mouse;
    std::shared_ptr<controller> board;
    std::shared_ptr<moon::rayTracingGraphics::RayTracingGraphics> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;
#endif

    bool enableBB{true};
    bool primitivesBB{false};
    bool treeBB{true};
    bool onlyLeafsBB{false};
    bool enableBloom{true};

    std::unordered_map<std::string, cuda::model> models;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);

public:
    testCuda(moon::graphicsManager::GraphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath, bool& framebufferResized);
    ~testCuda() = default;
    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTPOS_H
