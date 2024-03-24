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

class graphicsManager;
class rayTracingGraphics;
class imguiGraphics;

class testCuda : public scene
{
private:
    bool& framebufferResized;

    std::filesystem::path   ExternalPath;
    vector<uint32_t,2>      extent{0};
    vector<double,2>        mousePos{0.0};

    float                   focus = 0.049f;
    std::string             screenshot;

    graphicsManager *app{nullptr};
    GLFWwindow* window{nullptr};
    cuda::devicep<cuda::camera> cam;
    cuda::camera hostcam = cuda::camera(cuda::ray(cuda::vec4f(2.0f, 0.0f, 2.0f, 1.0f), cuda::vec4f(-1.0f, 0.0f, -1.0f, 0.0f)), 1.0f);

    std::shared_ptr<controller> mouse;
    std::shared_ptr<controller> board;
    std::shared_ptr<rayTracingGraphics> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<imguiGraphics> gui;
#endif

    bool enableBB{true};

    std::unordered_map<std::string, cuda::model> models;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);

public:
    testCuda(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath, bool& framebufferResized);
    ~testCuda() = default;
    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTPOS_H
