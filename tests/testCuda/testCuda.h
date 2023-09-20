#ifndef TESTPOS_H
#define TESTPOS_H

#include <glfw3.h>

#include <filesystem>
#include <vector>

#include "scene.h"
#include "controller.h"

#include "ray.h"

class graphicsManager;
class rayTracingGraphics;
class hitableArray;
namespace cuda {class camera;}

class testCuda : public scene
{
private:
    std::filesystem::path ExternalPath;
    double mousePosX = 0, mousePosY = 0;
    ray viewRay = ray(vec4(2.0f, 0.0f, 2.0f, 1.0f), vec4(-1.0f, 0.0f, -1.0f, 0.0f));
    float focus = 0.049f;
    uint32_t width = 800, height = 800;

    graphicsManager *app{nullptr};
    GLFWwindow* window{nullptr};
    cuda::camera* cam{nullptr};
    hitableArray* array{nullptr};
    rayTracingGraphics* graphics{nullptr};

    controller* mouse{nullptr};
    controller* board{nullptr};

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);

public:
    testCuda(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath);
    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
    void destroy() override;
};

#endif // TESTPOS_H
