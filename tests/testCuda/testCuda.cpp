#include "testCuda.h"

#include "rayTracingGraphics.h"
#include "graphicsManager.h"

#include "triangle.h"
#include "sphere.h"

#ifdef IMGUI_GRAPHICS
#include "imguiGraphics.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstring>

enum sign
{
    minus,
    plus
};

std::vector<cuda::vertex> createBoxVertexBuffer(vec4 scale, vec4 translate, sign normalSign, cuda::properties props, std::vector<vec4> colors) {
    float plus = normalSign == sign::plus ? 1.0f : -1.0f, minus = -plus;
    vec4 v[8] =
        {
            scale * vec4(-1.0f, -1.0f, -1.0f, 1.0f) + translate,
            scale * vec4(-1.0f,  1.0f, -1.0f, 1.0f) + translate,
            scale * vec4(1.0f, -1.0f, -1.0f, 1.0f) + translate,
            scale * vec4(1.0f,  1.0f, -1.0f, 1.0f) + translate,
            scale * vec4(-1.0f, -1.0f,  1.0f, 1.0f) + translate,
            scale * vec4(-1.0f,  1.0f,  1.0f, 1.0f) + translate,
            scale * vec4(1.0f, -1.0f,  1.0f, 1.0f) + translate,
            scale * vec4(1.0f,  1.0f,  1.0f, 1.0f) + translate
        };
    vec4 n[6] =
        {
            vec4(0.0f, 0.0f, minus, 0.0f), vec4(0.0f, 0.0f, plus, 0.0f), vec4(minus, 0.0f, 0.0f, 0.0f),
            vec4(plus, 0.0f, 0.0f, 0.0f), vec4(0.0f, minus, 0.0f, 0.0f), vec4(0.0f, plus, 0.0f, 0.0f)
        };
    size_t indices[6][4] = { {0,1,2,3}, {4,5,6,7}, {0,1,4,5}, {2,3,6,7}, {0,2,4,6}, {1,3,5,7} };

    std::vector<cuda::vertex> vertexBuffer;
    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 4; j++) {
            vertexBuffer.push_back(cuda::vertex(v[indices[i][j]], n[i], colors[i], props));
        }
    }
    return vertexBuffer;
}

std::vector<uint32_t> createBoxIndexBuffer() {
    return std::vector<uint32_t>{
        0, 1, 2, 3, 1, 2,
        4, 5, 6, 7, 5, 6,
        8, 9, 11, 10, 11, 8,
        12, 13, 15, 14, 15, 12,
        16, 17, 19, 16, 18, 19,
        20, 21, 23, 20, 22, 23
    };
}

void createWorld(std::unordered_map<std::string, cuda::model>& models)
{
    const std::unordered_map<std::string, cuda::sphere> spheres = {
        {"sphere_0",  cuda::sphere(vec4( 0.0f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(0.80f, 0.30f, 0.30f, 1.00f), { 1.0f, 0.0f, 0.0f, pi, 0.0f, 0.7f})},
        {"sphere_1",  cuda::sphere(vec4( 0.0f,  1.0f,  0.5f,  1.0f), 0.50f, vec4(0.80f, 0.80f, 0.80f, 1.00f), { 1.0f, 0.0f, 3.0f, 0.05f * pi, 0.0f, 0.7f})},
        {"sphere_2",  cuda::sphere(vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.50f, vec4(0.90f, 0.90f, 0.90f, 1.00f), { 1.5f, 0.96f, 0.001f, 0.0f, 0.0f, 1.0f})},
        {"sphere_3",  cuda::sphere(vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.45f, vec4(0.90f, 0.90f, 0.90f, 1.00f), { 1.0f / 1.5f, 0.96f, 0.001f, 0.0f, 0.0f, 1.0f})},
        {"sphere_4",  cuda::sphere(vec4(-1.5f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(1.00f, 0.90f, 0.70f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_5",  cuda::sphere(vec4( 1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4(0.99f, 0.80f, 0.20f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_6",  cuda::sphere(vec4( 1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4(0.20f, 0.80f, 0.99f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_7",  cuda::sphere(vec4(-1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4(0.99f, 0.40f, 0.85f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_8",  cuda::sphere(vec4(-1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4(0.40f, 0.99f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_9",  cuda::sphere(vec4(-0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4(0.65f, 0.00f, 0.91f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_10", cuda::sphere(vec4( 0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4(0.80f, 0.70f, 0.99f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_11", cuda::sphere(vec4(-0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4(0.59f, 0.50f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_12", cuda::sphere(vec4( 0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4(0.90f, 0.99f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_13", cuda::sphere(vec4(-1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4(0.65f, 0.00f, 0.91f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_14", cuda::sphere(vec4( 1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4(0.80f, 0.90f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_15", cuda::sphere(vec4(-1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4(0.90f, 0.50f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_16", cuda::sphere(vec4( 1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4(0.50f, 0.59f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})}
    };

    for(const auto& [name, sphere]: spheres){
        models[name] = cuda::model(cuda::primitive{cuda::make_devicep<cuda::hitable>(sphere), sphere.calcBox()});
    }

    const auto boxIndexBuffer = createBoxIndexBuffer();
    models["environment_box"] = cuda::model(
        createBoxVertexBuffer(
            vec4(3.0f, 3.0f, 1.5f, 1.0f),
            vec4(0.0f, 0.0f, 1.5f, 0.0f),
            sign::minus, { 1.0f, 0.0f, 0.0f, pi, 0.0f, 0.7f },
            { vec4(0.5f, 0.5f, 0.5f, 1.0f), vec4(0.5f, 0.5f, 0.5f, 1.0f), vec4(0.8f, 0.4f, 0.8f, 1.0f), vec4(0.4f, 0.4f, 0.4f, 1.0f), vec4(0.9f, 0.5f, 0.0f, 1.0f), vec4(0.1f, 0.4f, 0.9f, 1.0f) }),
            boxIndexBuffer);

    models["glass_box"] = cuda::model(
        createBoxVertexBuffer(
            vec4(0.4f, 0.4f, 0.4f, 1.0f),
            vec4(1.5f, 0.0f, 0.41f, 0.0f),
            sign::plus,
            { 1.5f, 1.0f, 0.01f, 0.01f * pi, 0.0f, 1.0f},
            std::vector<vec4>(6, vec4(1.0f))),
        boxIndexBuffer);

    models["glass_box_inside"] = cuda::model(
        createBoxVertexBuffer(
            vec4(0.3f, 0.3f, 0.3f, 1.0f),
            vec4(1.5f, 0.0f, 0.41f, 0.0f),
            sign::plus,
            { 1.0f / 1.5f, 1.0f, 0.01f, 0.01f * pi, 0.0f, 1.0f},
            std::vector<vec4>(6, vec4(1.0f))),
        boxIndexBuffer);

    models["upper_light_plane"] = cuda::model(
        createBoxVertexBuffer(
            vec4(2.0f, 2.0f, 0.01f, 1.0f),
            vec4(0.0f, 0.0f, 3.0f, 0.0f),
            sign::plus,
            { 0.0f, 0.0f, 0.0f, 0.0, 1.0f, 1.0f},
            std::vector<vec4>(6, vec4(1.0f))),
        boxIndexBuffer);

#if false
    for (int i = 0; i < 50; i++) {
        float phi = 2.0f * pi * static_cast<float>(i) / 50.0f;
        models["box_" + std::to_string(i)] = cuda::model(
            createBoxVertexBuffer(
                vec4(0.1f, 0.1f, 0.1f, 1.0f),
                vec4(2.8f * std::cos(phi), 2.8f * std::sin(phi), 0.1f, 0.0f),
                sign::plus,
                { 1.0f, 0.96f, std::sin(phi), std::abs(std::sin(phi) * std::cos(phi)) * pi, 0.0f },
                std::vector<vec4>(6, vec4(std::abs(std::cos(phi)), std::abs(std::sin(phi)), std::abs(std::sin(phi) * std::cos(phi)), 1.0f))),
            boxIndexBuffer);
    }
#endif
}

testCuda::testCuda(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath, bool& framebufferResized) :
    framebufferResized(framebufferResized),
    ExternalPath(ExternalPath),
    app(app),
    window(window),
    mouse(new controller(window, glfwGetMouseButton)),
    board(new controller(window, glfwGetKey))
{
    screenshot.resize(100);
    std::memcpy(screenshot.data(), "screenshot", 10);
}

void testCuda::create(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};
    createWorld(models);

    cam = cuda::make_devicep<cuda::camera>(cuda::camera(viewRay, float(extent[0]) / float(extent[1])));
    graphics = std::make_shared<rayTracingGraphics>(ExternalPath / "core/rayTracingGraphics/spv", VkExtent2D{extent[0],extent[1]});
    app->setGraphics(graphics.get());
    graphics->setCamera(&cam);
    graphics->setEnableBoundingBox(enableBB);
    graphics->create();
    for(auto& [name, model]: models){
        graphics->bind(&model);

#ifdef false
        std::cout << name << std::endl;
        for(const auto& primitive: model.primitives){
            const auto& box = primitive.box;
            std::cout << "--------------------------------\n";
            std::cout << "box " << &primitive - &model.primitives[0] << std::endl;
            std::cout << "min:\t" << box.min.x() << "\t" << box.min.y() << "\t" << box.min.z() << "\n";
            std::cout << "max:\t" << box.max.x() << "\t" << box.max.y() << "\t" << box.max.z() << "\n";
        }
        std::cout << "===============================\n";
#endif
    }

#ifdef IMGUI_GRAPHICS
    gui = std::make_shared<imguiGraphics>(window, app->getInstance(), app->getImageCount());
    app->setGraphics(gui.get());
    gui->create();
#endif
}

void testCuda::resize(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};
    cam = std::move(cuda::make_devicep<cuda::camera>(cuda::camera(viewRay, float(extent[0]) / float(extent[1]))));
    graphics->setExtent({extent[0],extent[1]});

    graphics->destroy();
    graphics->setEnableBoundingBox(enableBB);
    graphics->create();
}

void testCuda::updateFrame(uint32_t, float frameTime)
{
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
        cuda::camera hostcam = cuda::to_host(cam);
        hostcam.focus = focus;
        cuda::to_device(hostcam, cam);
        graphics->clearFrame();
    }

    cuda::camera hostCam = cuda::to_host(cam);
    vec4 o = hostCam.viewRay.getOrigin();
    std::string camPos = std::to_string(o.x()) + " " + std::to_string(o.y()) + " " + std::to_string(o.z());
    ImGui::Text("%s", camPos.c_str());

    if(ImGui::RadioButton("enable BB", enableBB)){
        enableBB = !enableBB;
        framebufferResized = true;
    }

    ImGui::End();

#else
    mouseEvent(frameTime);
    keyboardEvent(frameTime);
#endif
}

void testCuda::mouseEvent(float)
{
    float sensitivity = 0.02f;

    if(double x = 0, y = 0; mouse->pressed(GLFW_MOUSE_BUTTON_LEFT)){
        glfwGetCursorPos(window,&x,&y);
        float cos_delta = std::cos(sensitivity * static_cast<float>(mousePos[0] - x));
        float sin_delta = std::sin(sensitivity * static_cast<float>(mousePos[0] - x));
        viewRay = ray(  viewRay.getOrigin(),
                        vec4(   viewRay.getDirection().x() * cos_delta - viewRay.getDirection().y() * sin_delta,
                                viewRay.getDirection().y() * cos_delta + viewRay.getDirection().x() * sin_delta,
                                viewRay.getDirection().z() + sensitivity *static_cast<float>(mousePos[1] - y),
                                0.0f));
        cuda::camera hostcam = cuda::to_host(cam);
        hostcam.viewRay = viewRay;
        cuda::to_device(hostcam, cam);
        graphics->clearFrame();
        mousePos = {x,y};
    } else {
        glfwGetCursorPos(window,&mousePos[0],&mousePos[1]);
    }
}

void testCuda::keyboardEvent(float)
{
    float sensitivity = 0.1f;

    auto moveCamera = [&sensitivity, this](vec4 deltaOrigin){
        viewRay = ray(viewRay.getOrigin() + sensitivity * deltaOrigin, viewRay.getDirection());
        cuda::camera hostcam = cuda::to_host(cam);
        hostcam.viewRay = viewRay;
        cuda::to_device(hostcam, cam);
        graphics->clearFrame();
    };

    if(board->pressed(GLFW_KEY_W)) moveCamera( viewRay.getDirection());
    if(board->pressed(GLFW_KEY_S)) moveCamera(-viewRay.getDirection());
    if(board->pressed(GLFW_KEY_D)) moveCamera( vec4::getHorizontal(viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_A)) moveCamera(-vec4::getHorizontal(viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_X)) moveCamera( vec4::getVertical(viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_Z)) moveCamera(-vec4::getVertical(viewRay.getDirection()));

    if(board->released(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window,GLFW_TRUE);
}
