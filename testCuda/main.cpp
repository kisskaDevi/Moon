#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <sstream>
#include <utility>
#include <filesystem>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION

#include <stb_image.h>
#include <glfw3.h>

#ifdef TEST_CUDA
#include "testCuda.h"
#include "rayTracingGraphics.h"
#endif

#include "graphicsManager.h"
#include "hitableArray.h"

class controller
{
private:
    std::unordered_map<uint32_t, bool> keysMap;
    GLFWwindow* window{nullptr};
    int (*glfwGetFunction)(GLFWwindow*,int){nullptr};

public:
    float sensitivity{0.1f};

public:
    controller() = default;
    controller(GLFWwindow* window, int (*glfwGetFunction)(GLFWwindow*,int));
    bool pressed(uint32_t key);
    bool released(uint32_t key);
};

controller::controller(GLFWwindow* window, int (*glfwGetFunction)(GLFWwindow*,int)) : window(window), glfwGetFunction(glfwGetFunction){}

bool controller::pressed(uint32_t key){
    bool res = glfwGetFunction(window,key) == GLFW_PRESS;
    return res;
}

bool controller::released(uint32_t key){
    bool res = keysMap.count(key) > 0 && keysMap[key] == GLFW_PRESS && glfwGetFunction(window,key) == GLFW_RELEASE;
    keysMap[key] = glfwGetFunction(window,key);
    return res;
}

bool framebufferResized = false;
double mousePosX = 0, mousePosY = 0;
ray viewRay = ray(vec4(2.0f, 0.0f, 2.0f, 1.0f), vec4(-1.0f, 0.0f, -1.0f, 0.0f));

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName = "");
std::pair<uint32_t,uint32_t> resize(GLFWwindow* window, graphicsManager* app, rayTracingGraphics* graphics, cuda::camera* cam);
void keyboardEvent(controller& board, GLFWwindow* window, rayTracingGraphics* graphics, cuda::camera* cam);
void mouseEvent(controller& mouse, GLFWwindow* window, rayTracingGraphics* graphics, cuda::camera* cam);

int main()
{
    uint32_t width = 800, height = 800;
    const std::filesystem::path ExternalPath = std::filesystem::absolute(std::string(__FILE__) + "/../../");

    GLFWwindow* window = initializeWindow(width, height, ExternalPath / "dependences/texture/icon.png");
    controller board(window, glfwGetKey);
    controller mouse(window, glfwGetMouseButton);

    graphicsManager app;
    debug::checkResult(app.createSurface(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createDevice(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createSwapChain(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createLinker(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createSyncObjects(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));


#ifdef TEST_CUDA
    cuda::camera* cam = cuda::camera::create(viewRay, float(width) / float(height));
    hitableArray* array = hitableArray::create();

    std::vector<primitive> primitives;

    createWorld(primitives, array);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    rayTracingGraphics graphics(ExternalPath / "core/cudaRayTracing/rayTracingGraphics/spv",{width,height},{0,0});
    app.setGraphics(&graphics);
    graphics.setCamera(cam);
    graphics.createGraphics();
    graphics.setList(array);
#endif

    static auto pastTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        mouseEvent(mouse, window, &graphics, cam);
        keyboardEvent(board, window, &graphics, cam);

        float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::high_resolution_clock::now() - pastTime).count();
        pastTime = std::chrono::high_resolution_clock::now();
        glfwSetWindowTitle(window, std::stringstream("Vulkan [" + std::to_string(1.0f/frameTime) + " FPS]").str().c_str());

        if(app.checkNextFrame() != VK_ERROR_OUT_OF_DATE_KHR)
        {
            if (VkResult result = app.drawFrame(); result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
                std::tie(width, height) = resize(window,&app,&graphics, cam);
            } else if(result) {
                throw std::runtime_error("failed to with " + std::to_string(result));
            }
        }
    }

    debug::checkResult(app.deviceWaitIdle(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));


    graphics.destroyGraphics();
    hitableArray::destroy(array);
    cuda::camera::destroy(cam);

    app.destroySwapChain();
    app.destroyLinker();
    app.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

std::pair<uint32_t,uint32_t> resize(GLFWwindow* window, graphicsManager* app, rayTracingGraphics* graphics, cuda::camera* cam)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width * height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    app->deviceWaitIdle();
    app->destroySwapChain();
    app->destroyLinker();
    app->createSwapChain(window);
    app->createLinker();

    cuda::camera::destroy(cam);
    cam = cuda::camera::create(viewRay, float(width) / float(height));

    graphics->setCamera(cam);
    graphics->destroyGraphics();
    graphics->createGraphics();

    framebufferResized = false;

    return std::pair<uint32_t,uint32_t>(width, height);
}

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int){ framebufferResized = true;});

    if(iconName.string().size() > 0){
        int width, height, comp;
        stbi_uc* img = stbi_load(iconName.string().c_str(), &width, &height, &comp, 0);
        GLFWimage images{width,height,img};
        glfwSetWindowIcon(window,1,&images);
    }

    return window;
}

void mouseEvent(controller& mouse, GLFWwindow* window, rayTracingGraphics* graphics, cuda::camera* cam)
{
    float sensitivity = 0.02f;

    if(double x = 0, y = 0; mouse.pressed(GLFW_MOUSE_BUTTON_LEFT)){
        glfwGetCursorPos(window,&x,&y);
        float cos_delta = std::cos(sensitivity * static_cast<float>(mousePosX - x));
        float sin_delta = std::sin(sensitivity * static_cast<float>(mousePosX - x));
        viewRay = ray(
            viewRay.getOrigin(),
            vec4(
                viewRay.getDirection().x() * cos_delta - viewRay.getDirection().y() * sin_delta,
                viewRay.getDirection().y() * cos_delta + viewRay.getDirection().x() * sin_delta,
                viewRay.getDirection().z() + sensitivity *static_cast<float>(mousePosY - y),
                0.0f
            )
        );
        cuda::camera::setViewRay(cam, viewRay);
        graphics->update();

        mousePosX = x;
        mousePosY = y;
    } else {
        glfwGetCursorPos(window,&mousePosX,&mousePosY);
    }
}

void keyboardEvent(controller& board, GLFWwindow* window, rayTracingGraphics* graphics, cuda::camera* cam)
{
    float sensitivity = 0.1f;

    if(board.pressed(GLFW_KEY_W)){
        viewRay = ray(viewRay.getOrigin() + sensitivity * viewRay.getDirection(), viewRay.getDirection());
        cuda::camera::setViewRay(cam, viewRay);
        graphics->update();
    }
    if(board.pressed(GLFW_KEY_S)){
        viewRay = ray(viewRay.getOrigin() - sensitivity * viewRay.getDirection(), viewRay.getDirection());
        cuda::camera::setViewRay(cam, viewRay);
        graphics->update();
    }
    if(board.pressed(GLFW_KEY_D)){
        viewRay = ray(viewRay.getOrigin() + sensitivity * vec4::getHorizontal(viewRay.getDirection()), viewRay.getDirection());
        cuda::camera::setViewRay(cam, viewRay);
        graphics->update();
    }
    if(board.pressed(GLFW_KEY_A)){
        viewRay = ray(viewRay.getOrigin() - sensitivity * vec4::getHorizontal(viewRay.getDirection()), viewRay.getDirection());
        cuda::camera::setViewRay(cam, viewRay);
        graphics->update();
    }
    if(board.pressed(GLFW_KEY_X)){
        viewRay = ray(viewRay.getOrigin() + sensitivity * vec4::getVertical(viewRay.getDirection()), viewRay.getDirection());
        cuda::camera::setViewRay(cam, viewRay);
        graphics->update();
    }
    if(board.pressed(GLFW_KEY_Z)){
        viewRay = ray(viewRay.getOrigin() - sensitivity * vec4::getVertical(viewRay.getDirection()), viewRay.getDirection());
        cuda::camera::setViewRay(cam, viewRay);
        graphics->update();
    }

    if(board.released(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window,GLFW_TRUE);
}

