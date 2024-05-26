#include "testCuda.h"

#include "rayTracingGraphics.h"
#include "graphicsManager.h"

#include "sphere.h"

#ifdef IMGUI_GRAPHICS
#include "imguiGraphics.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif

#include <models/objmodel.h>

#include <cstring>
#include <random>

#include <math/quat2.h>
#include <math/mat4.h>
#include "transformational/camera.h"
#include "transformational/object.h"

using namespace cuda::rayTracing;
float pi = M_PI;

void createWorld(std::unordered_map<std::string, std::unique_ptr<cuda::rayTracing::Object>>& objects, const std::filesystem::path& ExternalPath)
{
    {
        std::unordered_map<std::string, Sphere> spheres = {
            {"glass_sphere_outside",  Sphere(vec4f( 1.5f, 0.0f,  0.51f,  1.0f), 0.50f, vec4f(0.90f, 0.90f, 0.90f, 1.00f), { 1.5f, 0.96f, 0.001f, 0.0f, 0.0f, 0.99f})},
            {"glass_sphere_inside",  Sphere(vec4f( 1.5f, 0.0f,  0.51f,  1.0f), 0.45f, vec4f(0.90f, 0.90f, 0.90f, 1.00f), { 1.0f / 1.5f, 0.96f, 0.001f, 0.0f, 0.0f, 0.99f})}
        };

        std::random_device dev;
        std::uniform_real_distribution<float> color(0.0f, 1.0f);
        std::uniform_real_distribution<float> xyz(-3.0f, 3.0f);
        std::uniform_real_distribution<float> power(8.0f, 10.0f);
        for(int i = 0 ; i < 100; i++){
            spheres.insert({"sphere_" + std::to_string(i),
             Sphere(
                 vec4f(xyz(dev), xyz(dev), std::abs(xyz(dev)),  1.0f),
                 0.02f,
                 vec4f(color(dev), color(dev), color(dev), 1.00f),
                 {0.0f, 0.0f, 0.0f, 0.0f, power(dev)})
            });
        }

        for(const auto& [name, sphere]: spheres){
            objects[name] = std::make_unique<Object>(
                new Model(Primitive{make_devicep<Hitable>(sphere), sphere.getBox()})
            );
        }
    }

    {
        objects["teapot"] = std::make_unique<Object>(
            new ObjModel(ExternalPath / "dependences/model/obj/teapot/teapot.obj",
                         ObjModelInfo(Properties{ 1.0f, 0.0f, 3.0f, 0.05f * pi, 0.0f, 0.7f}, vec4f(0.80f, 0.80f, 0.80f, 1.00f))),
            trans(vec4f(0.0f, 1.5f, 0.51f, 1.0f)) * toMat(quatf(0.5f *pi, vec4f{1.0f, 0.0f, 0.0f, 0.0})) * scale(vec4f{0.05f}));

        objects["environment_box"] = std::make_unique<Object>(
            new ObjModel(ExternalPath / "dependences/model/obj/box/box_in.obj",
                         ObjModelInfo(Properties{ 1.0f, 0.0f, 0.0f, pi, 0.0f, 0.7f }, vec4f{1.0f})),
            trans(vec4f{0.0f, 0.0f, 1.5f, 0.0f}) * scale(vec4f{3.0f, 3.0f, 1.5f, 1.0f}));

        objects["upper_light_plane"] = std::make_unique<Object>(
            new ObjModel(ExternalPath / "dependences/model/obj/box/box.obj",
                         ObjModelInfo(Properties{ 0.0f, 0.0f, 0.0f, 0.0, 10.0f, 1.0f}, vec4f{1.0f})),
            trans(vec4f{0.0f, 0.0f, 3.0f, 0.0f}) * scale(vec4f{2.0f, 2.0f, 0.01f, 1.0f}));

        objects["monkey"] = std::make_unique<Object>(
            new ObjModel(ExternalPath / "dependences/model/obj/monkey/monkey.obj",
                         ObjModelInfo(Properties{ 1.0f, 0.0f, 0.0f, pi, 0.0f, 0.7f}, vec4f(0.80f, 0.30f, 0.30f, 1.00f), true)),
            trans(vec4f{0.0f, 0.0f, 0.5f, 0.0f}) * toMat(quatf(0.5f * pi, vec4f{0.5f, 0.0f, 1.0f, 0.0f})) * scale(vec4f{0.5f}));

        objects["duck"] = std::make_unique<Object>(
            new ObjModel(ExternalPath / "dependences/model/obj/duck/duck.obj",
                         ObjModelInfo(Properties{ 1.0f, 0.0f, 0.0f, 0.5f * pi, 0.0f, 0.7f}, vec4f{0.8f, 0.8f, 0.0f, 1.0f}, true)),
            trans(vec4f{0.0f, -1.5f, 1.0f, 0.0f}) * toMat(quatf(0.5f *pi, vec4f{1.0f, 0.0f, 0.0f, 0.0})) * scale(vec4f{0.7f}));
    }

    {
        vec4f tr[4] = {
            vec4f{2.0f, 2.0f, 0.02f, 0.0f},
            vec4f{-2.0f, 2.0f, 0.02f, 0.0f},
            vec4f{2.0f, -2.0f, 0.02f, 0.0f},
            vec4f{-2.0f, -2.0f, 0.02f, 0.0f}
        };
        vec4f col[4] = {
            vec4f{0.2f, 0.9f, 0.4f, 1.0f},
            vec4f{0.9f, 0.5f, 0.2f, 1.0f},
            vec4f{0.9f, 0.4f, 0.9f, 1.0f},
            vec4f{0.0f, 0.7f, 0.9f, 1.0f}
        };
        for(int i = 0; i < 4; i++){
            objects["corner_light_plane_" + std::to_string(i)] = std::make_unique<Object>(
                new ObjModel(ExternalPath / "dependences/model/obj/box/box.obj",
                             ObjModelInfo(Properties{ 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 1.0f}, col[i])),
                trans(tr[i]) * scale(vec4f{0.4f, 0.4f, 0.01f, 1.0f}));
        }
    }

    {
        size_t num = 50;
        for (int i = 0; i < num; i++) {
            float phi = 2.0f * pi * static_cast<float>(i) / static_cast<float>(num);
            objects["box_" + std::to_string(i)] = std::make_unique<Object>(
                new ObjModel(ExternalPath / "dependences/model/obj/box/box.obj",
                             ObjModelInfo(
                                 Properties{0.0f,
                                            0.0f,
                                            std::sin(phi) * std::sin(phi),
                                            std::abs(std::sin(phi) + std::cos(phi)) * pi,
                                            0.0f,
                                            0.9},
                                          vec4f(std::abs(std::cos(phi)), std::abs(std::sin(phi)), 0.5f + 0.5f * std::sin(phi), 1.0f))),
                trans(vec4f(2.8f * std::cos(phi), 2.8f * std::sin(phi), 1.5f + 1.4f * std::sin(phi), 0.0f))
                    * toMat(quatf(phi, vec4f{std::cos(phi), std::sin(phi) * std::sin(phi), std::sin(phi) * std::cos(phi), 0.0f}))
                    * scale(vec4f{0.1f}));
        }
    }
}

testCuda::testCuda(moon::graphicsManager::GraphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath, bool& framebufferResized) :
    framebufferResized(framebufferResized),
    ExternalPath(ExternalPath),
    app(app),
    window(window),
    mouse(new controller(window, glfwGetMouseButton)),
    board(new controller(window, glfwGetKey))
{
    screenshot.resize(100);
    std::memcpy(screenshot.data(), "screenshot", 10);
    board->sensitivity = 0.1f;
    mouse->sensitivity = 0.02f;

    cam = Devicep<cuda::rayTracing::Camera>();

    hostcam = std::make_unique<cuda::rayTracing::Camera>(
        cuda::rayTracing::Camera(
            cuda::rayTracing::ray(
                cuda::rayTracing::vec4f(2.0f, 0.0f, 2.0f, 1.0f),
                cuda::rayTracing::vec4f(-1.0f, 0.0f, -1.0f, 0.0f)
                ),
            1.0f));
}

testCuda::~testCuda() = default;

void testCuda::create(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};
    createWorld(objects, ExternalPath);

    graphics = std::make_shared<moon::rayTracingGraphics::RayTracingGraphics>(
        ExternalPath / "core/rayTracingGraphics/spv",
        ExternalPath / "core/workflows/spv",
        VkExtent2D{extent[0],extent[1]});
    app->setGraphics(graphics.get());
    graphics->create();

    for(auto& [name, object]: objects){
        graphics->bind(object.get());
    }
    graphics->buildTree();

    hostcam->aspect = float(extent[0]) / float(extent[1]);
    cam = make_devicep<Camera>(*hostcam);

    graphics->buildBoundingBoxes(false, true, false);
    graphics->setBlitFactor(blitFactor);
    graphics->setCamera(&cam);
    graphics->setEnableBoundingBox(enableBB);

#ifdef IMGUI_GRAPHICS
    gui = std::make_shared<moon::imguiGraphics::ImguiGraphics>(window, app->getInstance(), app->getImageCount());
    app->setGraphics(gui.get());
    gui->create();
#endif
}

void testCuda::resize(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};
    hostcam->aspect = float(extent[0]) / float(extent[1]);
    cam = make_devicep<Camera>(*hostcam);
    graphics->setExtent({extent[0],extent[1]});

    graphics->destroy();
    graphics->setEnableBoundingBox(enableBB);
    graphics->setEnableBloom(enableBloom);
    graphics->create();
    graphics->setBlitFactor(blitFactor);
}

void testCuda::updateFrame(uint32_t, float frameTime)
{
    *hostcam = to_host(cam);
    glfwPollEvents();

#ifdef IMGUI_GRAPHICS
    ImGuiIO io = ImGui::GetIO();
    if(!io.WantCaptureMouse)    mouseEvent(frameTime);
    if(!io.WantCaptureKeyboard) keyboardEvent(frameTime);

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::SetWindowSize({350,100}, ImGuiCond_::ImGuiCond_Once);

    ImGui::Begin("Debug");

    if (ImGui::Button("Update")){
        framebufferResized = true;
    }

    ImGui::SameLine(0.0, 10.0f);
    if(ImGui::Button("Make screenshot")){
        const auto& imageExtent = app->getImageExtent();
        auto screenshot = app->makeScreenshot();

        std::vector<uint8_t> jpg(3 * imageExtent.height * imageExtent.width, 0);
        for (size_t pixel_index = 0, jpg_index = 0; pixel_index < imageExtent.height * imageExtent.width; pixel_index++) {
            jpg[jpg_index++] = static_cast<uint8_t>((screenshot[pixel_index] & 0x00ff0000) >> 16);
            jpg[jpg_index++] = static_cast<uint8_t>((screenshot[pixel_index] & 0x0000ff00) >> 8);
            jpg[jpg_index++] = static_cast<uint8_t>((screenshot[pixel_index] & 0x000000ff) >> 0);
        }
        auto filename = std::string("./") + std::string(this->screenshot.data()) + std::string(".jpg");
        stbi_write_jpg(filename.c_str(), imageExtent.width, imageExtent.height, 3, jpg.data(), 100);
    }

    ImGui::SameLine(0.0, 10.0f);
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputText("filename", screenshot.data(), screenshot.size());

    std::string title = "FPS = " + std::to_string(1.0f / frameTime);
    ImGui::Text("%s", title.c_str());

    if(ImGui::SliderFloat("focus", &focus, 0.03f, 0.1f, "%.5f")){
        hostcam->focus = focus;
        graphics->clearFrame();
    }

    if(ImGui::SliderFloat("bloom factor", &blitFactor, 1.0f, 3.0f)){
        graphics->setBlitFactor(blitFactor);
    }

    vec4f o = hostcam->viewRay.getOrigin();
    std::string camPos = std::to_string(o.x()) + " " + std::to_string(o.y()) + " " + std::to_string(o.z());
    ImGui::Text("%s", camPos.c_str());

    if(ImGui::RadioButton("bloom", enableBloom)){
        enableBloom = !enableBloom;
        framebufferResized = true;
    }

    if(ImGui::RadioButton("primitives BB", primitivesBB)){
        primitivesBB = !primitivesBB;
        graphics->buildBoundingBoxes(primitivesBB, treeBB, onlyLeafsBB);
    }

    if(ImGui::RadioButton("tree BB", treeBB)){
        treeBB = !treeBB;
        graphics->buildBoundingBoxes(primitivesBB, treeBB, onlyLeafsBB);
    }

    if(ImGui::RadioButton("only leafs BB", onlyLeafsBB)){
        onlyLeafsBB = !onlyLeafsBB;
        graphics->buildBoundingBoxes(primitivesBB, treeBB, onlyLeafsBB);
    }

    ImGui::End();

#else
    mouseEvent(frameTime);
    keyboardEvent(frameTime);
#endif
    to_device(*hostcam, cam);
}

void testCuda::mouseEvent(float frameTime)
{
    double x = 0, y = 0;
    glfwGetCursorPos(window,&x,&y);
    if(mouse->pressed(GLFW_MOUSE_BUTTON_LEFT)){
        const float ms = mouse->sensitivity * frameTime * 40;
        float dcos = std::cos(ms * static_cast<float>(mousePos[0] - x));
        float dsin = std::sin(ms * static_cast<float>(mousePos[0] - x));
        float dz = ms * static_cast<float>(mousePos[1] - y);
        const vec4f& d = hostcam->viewRay.getDirection();
        const vec4f& o = hostcam->viewRay.getOrigin();
        hostcam->viewRay = ray(o, vec4f(d.x() * dcos - d.y() * dsin, d.y() * dcos + d.x() * dsin, d.z() + dz, 0.0f));

        graphics->clearFrame();
    }
    mousePos = {x,y};
}

void testCuda::keyboardEvent(float frameTime)
{
    auto moveCamera = [this](vec4f deltaOrigin){
        hostcam->viewRay = ray(hostcam->viewRay.getOrigin() + deltaOrigin, hostcam->viewRay.getDirection());
        graphics->clearFrame();
    };

    const float bs = board->sensitivity * frameTime * 40;
    if(board->pressed(GLFW_KEY_W)) moveCamera( bs * hostcam->viewRay.getDirection());
    if(board->pressed(GLFW_KEY_S)) moveCamera(-bs * hostcam->viewRay.getDirection());
    if(board->pressed(GLFW_KEY_D)) moveCamera( bs * vec4f::getHorizontal(hostcam->viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_A)) moveCamera(-bs * vec4f::getHorizontal(hostcam->viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_X)) moveCamera( bs * vec4f::getVertical(hostcam->viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_Z)) moveCamera(-bs * vec4f::getVertical(hostcam->viewRay.getDirection()));

    if(board->released(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window,GLFW_TRUE);
}
