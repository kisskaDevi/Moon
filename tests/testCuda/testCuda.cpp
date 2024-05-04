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

using namespace cuda;

enum sign{ minus, plus };

std::vector<cuda::vertex> createBoxVertexBuffer(vec4f scale, vec4f translate, sign normalSign, cuda::properties props, std::vector<vec4f> colors) {
    float plus = normalSign == sign::plus ? 1.0f : -1.0f, minus = -plus;
    vec4f v[8] =
        {
            scale * vec4f(-1.0f, -1.0f, -1.0f, 1.0f) + translate,
            scale * vec4f(-1.0f,  1.0f, -1.0f, 1.0f) + translate,
            scale * vec4f(1.0f, -1.0f, -1.0f, 1.0f) + translate,
            scale * vec4f(1.0f,  1.0f, -1.0f, 1.0f) + translate,
            scale * vec4f(-1.0f, -1.0f,  1.0f, 1.0f) + translate,
            scale * vec4f(-1.0f,  1.0f,  1.0f, 1.0f) + translate,
            scale * vec4f(1.0f, -1.0f,  1.0f, 1.0f) + translate,
            scale * vec4f(1.0f,  1.0f,  1.0f, 1.0f) + translate
        };
    vec4f n[6] =
        {
            vec4f(0.0f, 0.0f, minus, 0.0f), vec4f(0.0f, 0.0f, plus, 0.0f), vec4f(minus, 0.0f, 0.0f, 0.0f),
            vec4f(plus, 0.0f, 0.0f, 0.0f), vec4f(0.0f, minus, 0.0f, 0.0f), vec4f(0.0f, plus, 0.0f, 0.0f)
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
        {"sphere_0",  cuda::sphere(vec4f( 0.0f,  0.0f,  0.5f,  1.0f), 0.50f, vec4f(0.80f, 0.30f, 0.30f, 1.00f), { 1.0f, 0.0f, 0.0f, pi, 0.0f, 0.7f})},
        {"sphere_1",  cuda::sphere(vec4f( 0.0f,  1.0f,  0.5f,  1.0f), 0.50f, vec4f(0.80f, 0.80f, 0.80f, 1.00f), { 1.0f, 0.0f, 3.0f, 0.05f * pi, 0.0f, 0.7f})},
        {"sphere_2",  cuda::sphere(vec4f( 0.0f, -1.0f,  0.5f,  1.0f), 0.50f, vec4f(0.90f, 0.90f, 0.90f, 1.00f), { 1.5f, 0.96f, 0.001f, 0.0f, 0.0f, 0.99f})},
        {"sphere_3",  cuda::sphere(vec4f( 0.0f, -1.0f,  0.5f,  1.0f), 0.45f, vec4f(0.90f, 0.90f, 0.90f, 1.00f), { 1.0f / 1.5f, 0.96f, 0.001f, 0.0f, 0.0f, 0.99f})},
        {"sphere_4",  cuda::sphere(vec4f(-1.5f,  0.0f,  0.5f,  1.0f), 0.50f, vec4f(1.00f, 0.90f, 0.70f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_5",  cuda::sphere(vec4f( 1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4f(0.99f, 0.80f, 0.20f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_6",  cuda::sphere(vec4f( 1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4f(0.20f, 0.80f, 0.99f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_7",  cuda::sphere(vec4f(-1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4f(0.99f, 0.40f, 0.85f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_8",  cuda::sphere(vec4f(-1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4f(0.40f, 0.99f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_9",  cuda::sphere(vec4f(-0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4f(0.65f, 0.00f, 0.91f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_10", cuda::sphere(vec4f( 0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4f(0.80f, 0.70f, 0.99f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_11", cuda::sphere(vec4f(-0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4f(0.59f, 0.50f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_12", cuda::sphere(vec4f( 0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4f(0.90f, 0.99f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_13", cuda::sphere(vec4f(-1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4f(0.65f, 0.00f, 0.91f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_14", cuda::sphere(vec4f( 1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4f(0.80f, 0.90f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_15", cuda::sphere(vec4f(-1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4f(0.90f, 0.50f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})},
        {"sphere_16", cuda::sphere(vec4f( 1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4f(0.50f, 0.59f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})}
    };

    for(const auto& [name, sphere]: spheres){
        models[name] = cuda::model(cuda::primitive{cuda::make_devicep<cuda::hitable>(sphere), sphere.getBox()});
    }

    const auto boxIndexBuffer = createBoxIndexBuffer();

    models["environment_box"] = cuda::model(
        createBoxVertexBuffer(
            vec4f(3.0f, 3.0f, 1.5f, 1.0f),
            vec4f(0.0f, 0.0f, 1.5f, 0.0f),
            sign::minus,
            { 1.0f, 0.0f, 0.0f, pi, 0.0f, 0.7f },
            { vec4f(0.5f, 0.5f, 0.5f, 1.0f), vec4f(0.5f, 0.5f, 0.5f, 1.0f), vec4f(0.8f, 0.4f, 0.8f, 1.0f), vec4f(0.4f, 0.4f, 0.4f, 1.0f), vec4f(0.9f, 0.5f, 0.0f, 1.0f), vec4f(0.1f, 0.4f, 0.9f, 1.0f) }),
            boxIndexBuffer);

    models["glass_box"] = cuda::model(
        createBoxVertexBuffer(
            vec4f(0.4f, 0.4f, 0.4f, 1.0f),
            vec4f(1.5f, 0.0f, 0.41f, 0.0f),
            sign::plus,
            { 1.5f, 1.0f, 0.01f, 0.01f * pi, 0.0f, 0.99f},
            std::vector<vec4f>(6, vec4f(1.0f))),
        boxIndexBuffer);

    models["glass_box_inside"] = cuda::model(
        createBoxVertexBuffer(
            vec4f(0.3f, 0.3f, 0.3f, 1.0f),
            vec4f(1.5f, 0.0f, 0.41f, 0.0f),
            sign::plus,
            { 1.0f / 1.5f, 1.0f, 0.01f, 0.01f * pi, 0.0f, 0.99f},
            std::vector<vec4f>(6, vec4f(1.0f))),
        boxIndexBuffer);

    models["upper_light_plane"] = cuda::model(
        createBoxVertexBuffer(
            vec4f(2.0f, 2.0f, 0.01f, 1.0f),
            vec4f(0.0f, 0.0f, 3.0f, 0.0f),
            sign::plus,
            { 0.0f, 0.0f, 0.0f, 0.0, 1.0f, 1.0f},
            std::vector<vec4f>(6, vec4f(1.0f))),
        boxIndexBuffer);

#if 1
    size_t num = 100;
    for (int i = 0; i < num; i++) {
        float phi = 2.0f * pi * static_cast<float>(i) / static_cast<float>(num);
        models["box_" + std::to_string(i)] = cuda::model(
            createBoxVertexBuffer(
                vec4f(0.1f, 0.1f, 0.1f, 1.0f),
                vec4f(2.8f * std::cos(phi), 2.8f * std::sin(phi), 0.1f + 2.8 * std::abs(std::sin(phi)), 0.0f),
                sign::plus,
                { 0.0f, 0.0f, std::sin(phi), std::abs(std::sin(phi) * std::cos(phi)) * pi, 0.0f, 0.9},
                std::vector<vec4f>(6, vec4f(std::abs(std::cos(phi)), std::abs(std::sin(phi)), std::abs(std::sin(phi) * std::cos(phi)), 1.0f))),
            boxIndexBuffer);
    }
#endif
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
}

void testCuda::create(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};
    createWorld(models);

    hostcam = cuda::camera(hostcam.viewRay, float(extent[0]) / float(extent[1]));
    cam = cuda::make_devicep<cuda::camera>(hostcam);
    graphics = std::make_shared<rayTracingGraphics>(ExternalPath / "core/rayTracingGraphics/spv", ExternalPath / "core/workflows/spv", VkExtent2D{extent[0],extent[1]});
    app->setGraphics(graphics.get());
    graphics->setCamera(&cam);
    graphics->setEnableBoundingBox(enableBB);
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
    graphics->create();
    graphics->buildTree();
    graphics->buildBoundingBoxes(false, true, false);

#ifdef IMGUI_GRAPHICS
    gui = std::make_shared<moon::imguiGraphics::ImguiGraphics>(window, app->getInstance(), app->getImageCount());
    app->setGraphics(gui.get());
    gui->create();
#endif
}

void testCuda::resize(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};
    hostcam = cuda::camera(hostcam.viewRay, float(extent[0]) / float(extent[1]));
    cam = std::move(cuda::make_devicep<cuda::camera>(hostcam));
    graphics->setExtent({extent[0],extent[1]});

    graphics->destroy();
    graphics->setEnableBoundingBox(enableBB);
    graphics->setEnableBloom(enableBloom);
    graphics->create();
}

void testCuda::updateFrame(uint32_t, float frameTime)
{
    hostcam = cuda::to_host(cam);
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
        hostcam.focus = focus;
        graphics->clearFrame();
    }

    if(ImGui::SliderFloat("bloom factor", &blitFactor, 1.0f, 3.0f)){
        graphics->setBlitFactor(blitFactor);
    }

    vec4f o = hostcam.viewRay.getOrigin();
    std::string camPos = std::to_string(o.x()) + " " + std::to_string(o.y()) + " " + std::to_string(o.z());
    ImGui::Text("%s", camPos.c_str());

    if(ImGui::RadioButton("bloom", enableBloom)){
        enableBloom = !enableBloom;
        framebufferResized = true;
    }

    if(ImGui::RadioButton("BB", enableBB)){
        enableBB = !enableBB;
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
    cuda::to_device(hostcam, cam);
}

void testCuda::mouseEvent(float)
{
    double x = 0, y = 0;
    glfwGetCursorPos(window,&x,&y);
    if(mouse->pressed(GLFW_MOUSE_BUTTON_LEFT)){
        float& ms = mouse->sensitivity;
        float dcos = std::cos(ms * static_cast<float>(mousePos[0] - x));
        float dsin = std::sin(ms * static_cast<float>(mousePos[0] - x));
        float dz = ms * static_cast<float>(mousePos[1] - y);
        const vec4f& d = hostcam.viewRay.getDirection();
        const vec4f& o = hostcam.viewRay.getOrigin();
        hostcam.viewRay = ray(o, vec4f(d.x() * dcos - d.y() * dsin, d.y() * dcos + d.x() * dsin, d.z() + dz, 0.0f));

        graphics->clearFrame();
    }
    mousePos = {x,y};
}

void testCuda::keyboardEvent(float)
{
    auto moveCamera = [this](vec4f deltaOrigin){
        hostcam.viewRay = ray(hostcam.viewRay.getOrigin() + deltaOrigin, hostcam.viewRay.getDirection());
        graphics->clearFrame();
    };

    const float& bs = board->sensitivity;
    if(board->pressed(GLFW_KEY_W)) moveCamera( bs * hostcam.viewRay.getDirection());
    if(board->pressed(GLFW_KEY_S)) moveCamera(-bs * hostcam.viewRay.getDirection());
    if(board->pressed(GLFW_KEY_D)) moveCamera( bs * vec4f::getHorizontal(hostcam.viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_A)) moveCamera(-bs * vec4f::getHorizontal(hostcam.viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_X)) moveCamera( bs * vec4f::getVertical(hostcam.viewRay.getDirection()));
    if(board->pressed(GLFW_KEY_Z)) moveCamera(-bs * vec4f::getVertical(hostcam.viewRay.getDirection()));

    if(board->released(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window,GLFW_TRUE);
}
