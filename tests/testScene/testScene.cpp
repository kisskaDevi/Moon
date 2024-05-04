#include "testScene.h"
#include "deferredGraphics.h"
#include "graphicsManager.h"
#include "gltfmodel.h"
#include "plymodel.h"
#include "spotLight.h"
#include "baseObject.h"
#include "group.h"
#include "baseCamera.h"

#ifdef SECOND_VIEW_WINDOW
#include "dualQuaternion.h"
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef IMGUI_GRAPHICS
#include "imguiGraphics.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#endif

#include <random>
#include <limits>
#include <cstring>

testScene::testScene(moon::graphicsManager::GraphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath, bool& framebufferResized):
    framebufferResized(framebufferResized),
    ExternalPath(ExternalPath),
    window(window),
    app(app),
    mouse(new controller(window, glfwGetMouseButton)),
    board(new controller(window, glfwGetKey)),
    resourceCount(app->getResourceCount()),
    imageCount(app->getImageCount())
{
    screenshot.resize(100);
    std::memcpy(screenshot.data(), "screenshot", 10);
}

void testScene::resize(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};

    cameras["base"]->recreate(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["base"]->setExtent({WIDTH, HEIGHT});

#ifdef SECOND_VIEW_WINDOW
    cameras["view"]->recreate(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["view"]->setExtent({static_cast<uint32_t>(WIDTH / 3), static_cast<uint32_t>(HEIGHT / 3)});
#endif

    for(auto& [_,graph]: graphics){
        graph->destroy();
        graph->create();
    }
}

void testScene::create(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};

    cameras["base"] = std::make_shared<baseCamera>(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["base"] = std::make_shared<deferredGraphics>(ExternalPath / "core/deferredGraphics/spv", ExternalPath / "core/workflows/spv", VkExtent2D{WIDTH, HEIGHT});
    app->setGraphics(graphics["base"].get());
    graphics["base"]->bind(cameras["base"].get());
    graphics["base"]->
        setEnable("TransparentLayer", true).
        setEnable("Skybox", true).
        setEnable("Blur", true).
        setEnable("Bloom", true).
        setEnable("SSAO", false).
        setEnable("SSLR", false).
        setEnable("Scattering", true).
        setEnable("Shadow", true).
        setEnable("Selector", true);
    graphics["base"]->create();

#ifdef SECOND_VIEW_WINDOW
    cameras["view"] = std::make_shared<baseCamera>(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["view"] = std::make_shared<deferredGraphics>(ExternalPath / "core/deferredGraphics/spv", VkExtent2D{WIDTH/3, HEIGHT/3});
    graphics["view"]->setPositionInWindow(viewOffset, viewExtent);
    app->setGraphics(graphics["view"].get());
    graphics["view"]->bind(cameras["view"].get());
    graphics["view"]->
        setEnable("TransparentLayer", true).
        setEnable("Skybox", false).
        setEnable("Blur", false).
        setEnable("Bloom", false).
        setEnable("SSAO", false).
        setEnable("SSLR", false).
        setEnable("Scattering", false).
        setEnable("Shadow", false);
    graphics["view"]->create();
#endif

#ifdef IMGUI_GRAPHICS
    gui = std::make_shared<moon::imguiGraphics::ImguiGraphics>(window, app->getInstance(), app->getImageCount());
    app->setGraphics(gui.get());
    gui->create();
#endif

    loadModels();
    createObjects();
    createLight();

    groups["lightBox"]->translate({0.0f,0.0f,25.0f});
    groups["ufo0"]->translate({5.0f,0.0f,5.0f});
    groups["ufo1"]->translate({-5.0f,0.0f,5.0f});
    groups["ufo2"]->translate({10.0f,0.0f,5.0f});
    groups["ufo3"]->translate({-10.0f,0.0f,5.0f});
}

void testScene::updateFrame(uint32_t frameNumber, float frameTime)
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

    auto switcher = [this](std::shared_ptr<deferredGraphics> graphics, const std::string& name){
        if(auto val = graphics->getEnable(name); ImGui::RadioButton(name.c_str(), val)){
            graphics->setEnable(name, !val);
            framebufferResized = true;
        }
    };

    if (ImGui::TreeNodeEx("Props", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen))
    {
        std::string title = "FPS = " + std::to_string(1.0f / frameTime);
        ImGui::Text("%s", title.c_str());
        ImGui::SliderFloat("bloom", &blitFactor, 1.0f, 3.0f);

        if(graphics["base"]->getEnable("Blur")){
            ImGui::SliderFloat("farBlurDepth", &farBlurDepth, 0.0f, 1.0f);
        } else {
            farBlurDepth = 1.0f;
        }

        ImGui::SliderFloat("ambient", &minAmbientFactor, 0.0f, 1.0f);
        for(auto& [_,graph]: graphics){
            graph->setMinAmbientFactor(minAmbientFactor);
        }

        ImGui::SliderFloat("animation speed", &animationSpeed, 0.0f, 5.0f);

        if(ImGui::RadioButton("refraction of scattering", enableScatteringRefraction)){
            enableScatteringRefraction = !enableScatteringRefraction;
            for(auto& [_,graph]: graphics){
                graph->setScatteringRefraction(enableScatteringRefraction);
            }
        }

        switcher(graphics["base"], "Bloom");
        switcher(graphics["base"], "Blur");
        switcher(graphics["base"], "Skybox");
        switcher(graphics["base"], "SSLR");
        switcher(graphics["base"], "SSAO");
        switcher(graphics["base"], "Shadow");
        switcher(graphics["base"], "Scattering");
        switcher(graphics["base"], "BoundingBox");
        switcher(graphics["base"], "TransparentLayer");

        ImGui::TreePop();
    }

#ifdef SECOND_VIEW_WINDOW
    if (ImGui::TreeNodeEx("Second Window", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("v_offset_x", &viewOffset[0], 0.0f, 1.0f);
        ImGui::SliderFloat("v_offset_y", &viewOffset[1], 0.0f, 1.0f);
        ImGui::SliderFloat("v_extent_x", &viewExtent[0], 0.0f, 1.0f);
        ImGui::SliderFloat("v_extent_y", &viewExtent[1], 0.0f, 1.0f);
        graphics["view"]->setPositionInWindow(viewOffset, viewExtent);

        switcher(graphics["view"], "Bloom");
        switcher(graphics["view"], "Blur");
        switcher(graphics["view"], "Skybox");
        switcher(graphics["view"], "SSLR");
        switcher(graphics["view"], "SSAO");
        switcher(graphics["view"], "Shadow");
        switcher(graphics["view"], "Scattering");
        switcher(graphics["view"], "BoundingBox");
        switcher(graphics["view"], "TransparentLayer");
    }
#endif

    if (ImGui::TreeNodeEx("Object", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen))
    {
        std::string title = "controled object : " + controledObjectName;
        ImGui::Text("%s", title.c_str());
        {
            if(controledObject){
                const auto p = controledObject->getTranslation();
                std::string pos = "position : " + std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]);
                ImGui::Text("%s", pos.c_str());
                const auto r = controledObject->getRotation();
                std::string rot = "rot : " + std::to_string(r.re()) + " " + std::to_string(r.im()[0]) + " " + std::to_string(r.im()[1]) + " " + std::to_string(r.im()[2]);
                ImGui::Text("%s", rot.c_str());
                const auto s = controledObject->getScale();
                std::string scl = "scale : " + std::to_string(s[0]) + " " + std::to_string(s[1]) + " " + std::to_string(s[2]);
                ImGui::Text("%s", scl.c_str());
            }
        }
        if(ImGui::RadioButton("outlighting", controledObjectEnableOutlighting)){
            controledObjectEnableOutlighting = !controledObjectEnableOutlighting;
            if(controledObject){
                controledObject->setOutlining(controledObjectEnableOutlighting);
            }
        }
        if(ImGui::ColorPicker4("outlighting", controledObjectOutlightingColor, ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_DisplayRGB)){
            if(controledObject){
                controledObject->setOutlining(true && controledObjectEnableOutlighting, 0.03f,
                    {
                        controledObjectOutlightingColor[0],
                        controledObjectOutlightingColor[1],
                        controledObjectOutlightingColor[2],
                        controledObjectOutlightingColor[3]
                    }
                );
            }
        }
        for(auto& [_,graph]: graphics){
            graph->updateCmdFlags();
        }
        ImGui::TreePop();
    }

    ImGui::End();

#else
    mouseEvent(frameTime);
    keyboardEvent(frameTime);
#endif

    updates(frameTime);

    for(auto& [_,object]: objects){
        object->updateAnimation(frameNumber, animationSpeed * frameTime);
    }
}

void testScene::loadModels()
{
    models["bee"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glb/Bee.glb", 2 * resourceCount);
    models["ufo"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glb/RetroUFO.glb");
    models["box"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glTF-Sample-Models/2.0/Box/glTF-Binary/Box.glb");
    models["sponza"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf");
    models["duck"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glTF-Sample-Models/2.0/Duck/glTF-Binary/Duck.glb");
    models["DragonAttenuation"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glTF-Sample-Models/2.0/DragonAttenuation/glTF-Binary/DragonAttenuation.glb");
    models["DamagedHelmet"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glTF-Sample-Models/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb");
    models["robot"] = std::make_shared<gltfModel>(ExternalPath / "dependences/model/glb/Robot.glb", resourceCount);
    models["floor"] = std::make_shared<plyModel>(ExternalPath / "dependences/model/ply/cube.ply");

    for(auto& [_,model]: models){
        graphics["base"]->create(model.get());
    }
}

void testScene::createObjects()
{
    staticObjects["sponza"] = std::make_shared<baseObject>(models["sponza"].get());
    staticObjects["sponza"]->rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({3.0f,3.0f,3.0f});

    objects["bee0"] = std::make_shared<baseObject>(models["bee"].get(), 0, resourceCount);
    objects["bee0"]->translate({3.0f,0.0f,0.0f}).rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({0.2f,0.2f,0.2f});

    objects["bee1"] = std::make_shared<baseObject>(models["bee"].get(), resourceCount, resourceCount);
    objects["bee1"]->translate({-3.0f,0.0f,0.0f}).rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({0.2f,0.2f,0.2f}).setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.7f));
    objects["bee1"]->setAnimation(1, 1.0f);

    objects["duck"] = std::make_shared<baseObject>(models["duck"].get());
    objects["duck"]->translate({0.0f,0.0f,3.0f}).rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({3.0f});
    objects["duck"]->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.8f));

    objects["lightBox"] = std::make_shared<baseObject>(models["box"].get());
    objects["lightBox"]->setBloomColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    groups["lightBox"] = std::make_shared<group>();
    groups["lightBox"]->addObject(objects["lightBox"].get());

    objects["dragon"] = std::make_shared<baseObject>(models["DragonAttenuation"].get());
    objects["dragon"]->scale(1.0f).rotate(quaternion<float>(0.5f, 0.5f, -0.5f, -0.5f)).translate(vector<float,3>(26.0f, 11.0f, 11.0f));

    objects["helmet"] = std::make_shared<baseObject>(models["DamagedHelmet"].get());
    objects["helmet"]->scale(1.0f).rotate(quaternion<float>(0.5f, 0.5f, -0.5f, -0.5f)).translate(vector<float,3>(27.0f, -10.0f, 14.0f));

    objects["robot"] = std::make_shared<baseObject>(models["robot"].get(), 0, resourceCount);
    objects["robot"]->scale(25.0f).rotate(quaternion<float>(0.5f, 0.5f, -0.5f, -0.5f)).rotate(radians(180.0f), {0.0f, 0.0f, 1.0f}).translate(vector<float,3>(-30.0f, 11.0f, 10.0f));

    objects["ufo_light_0"] = std::make_shared<baseObject>(models["ufo"].get());
    objects["ufo_light_0"]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
    groups["ufo_light_0"] = std::make_shared<group>();
    groups["ufo_light_0"]->rotate(radians(45.0f),vector<float,3>(1.0f,0.0f,0.0f)).rotate(radians(45.0f),vector<float,3>(0.0f,0.0f,-1.0f)).translate(vector<float,3>(24.0f, 7.5f, 18.0f));
    groups["ufo_light_0"]->addObject(objects["ufo_light_0"].get());

    objects["ufo_light_1"] = std::make_shared<baseObject>(models["ufo"].get());
    objects["ufo_light_1"]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
    groups["ufo_light_1"] = std::make_shared<group>();
    groups["ufo_light_1"]->rotate(radians(45.0f),vector<float,3>(-1.0f,0.0f,0.0f)).rotate(radians(45.0f),vector<float,3>(0.0f,0.0f,1.0f)).translate(vector<float,3>(24.0f, -7.5f, 18.0f));
    groups["ufo_light_1"]->addObject(objects["ufo_light_1"].get());

    objects["ufo_light_2"] = std::make_shared<baseObject>(models["ufo"].get());
    objects["ufo_light_2"]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
    groups["ufo_light_2"] = std::make_shared<group>();
    groups["ufo_light_2"]->rotate(radians(30.0f),vector<float,3>(-1.0f,0.0f,0.0f)).rotate(radians(30.0f),vector<float,3>(0.0f,0.0f,1.0f)).translate(vector<float,3>(-32.0f, 13.0f, 19.0f));
    groups["ufo_light_2"]->addObject(objects["ufo_light_2"].get());

    objects["ufo_light_3"] = std::make_shared<baseObject>(models["ufo"].get());
    objects["ufo_light_3"]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
    groups["ufo_light_3"] = std::make_shared<group>();
    groups["ufo_light_3"]->rotate(radians(30.0f),vector<float,3>(1.0f,0.0f,0.0f)).rotate(radians(30.0f),vector<float,3>(0.0f,0.0f,-1.0f)).translate(vector<float,3>(-32.0f, 7.0f, 19.0f));
    groups["ufo_light_3"]->addObject(objects["ufo_light_3"].get());

    objects["ufo_light_4"] = std::make_shared<baseObject>(models["ufo"].get());
    objects["ufo_light_4"]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
    groups["ufo_light_4"] = std::make_shared<group>();
    groups["ufo_light_4"]->rotate(radians(30.0f),vector<float,3>(-1.0f,0.0f,0.0f)).rotate(radians(30.0f),vector<float,3>(0.0f,0.0f,-1.0f)).translate(vector<float,3>(-26.0f, 13.0f, 19.0f));
    groups["ufo_light_4"]->addObject(objects["ufo_light_4"].get());

    for(auto key = "ufo" + std::to_string(ufoCounter); ufoCounter < 4; ufoCounter++, key = "ufo" + std::to_string(ufoCounter)){
        objects[key] = std::make_shared<baseObject>(models["ufo"].get());
        objects[key]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
        objects[key]->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.8f));
        groups[key] = std::make_shared<group>();
        groups[key]->addObject(objects["ufo" + std::to_string(ufoCounter)].get());
    }

    skyboxObjects["lake"] = std::make_shared<skyboxObject>(
        std::vector<std::filesystem::path>{
            ExternalPath / "dependences/texture/skybox/left.jpg",
            ExternalPath / "dependences/texture/skybox/right.jpg",
            ExternalPath / "dependences/texture/skybox/front.jpg",
            ExternalPath / "dependences/texture/skybox/back.jpg",
            ExternalPath / "dependences/texture/skybox/top.jpg",
            ExternalPath / "dependences/texture/skybox/bottom.jpg"
    });
    skyboxObjects["lake"]->scale({200.0f,200.0f,200.0f});

    skyboxObjects["stars"] = std::make_shared<skyboxObject>(
        std::vector<std::filesystem::path>{
            ExternalPath / "dependences/texture/skybox1/left.png",
            ExternalPath / "dependences/texture/skybox1/right.png",
            ExternalPath / "dependences/texture/skybox1/front.png",
            ExternalPath / "dependences/texture/skybox1/back.png",
            ExternalPath / "dependences/texture/skybox1/top.png",
            ExternalPath / "dependences/texture/skybox1/bottom.png"
    });
    skyboxObjects["stars"]->scale({200.0f,200.0f,200.0f});

    for(auto& [_,graph]: graphics){
        for(auto& [_,object]: objects){
            graph->bind(object.get());
        }
        for(auto& [_,object]: staticObjects){
            graph->bind(object.get());
        }
        for(auto& [_, object]: skyboxObjects){
            graph->bind(object.get());
        }
    }
}

void testScene::createLight()
{
    std::filesystem::path LIGHT_TEXTURE0  = ExternalPath / "dependences/texture/icon.PNG";
    std::filesystem::path LIGHT_TEXTURE1  = ExternalPath / "dependences/texture/light1.jpg";
    std::filesystem::path LIGHT_TEXTURE2  = ExternalPath / "dependences/texture/light2.jpg";
    std::filesystem::path LIGHT_TEXTURE3  = ExternalPath / "dependences/texture/light3.jpg";

    lightPoints["lightBox"] = std::make_shared<isotropicLight>(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    groups["lightBox"]->addObject(lightPoints["lightBox"].get());

    for(const auto& light: lightPoints["lightBox"]->get()){
        lightSources.push_back(std::shared_ptr<spotLight>(light));
    }

    matrix<float,4,4> proj = perspective(radians(90.0f), 1.0f, 0.1f, 20.0f);

    lightSources.push_back(std::make_shared<spotLight>(LIGHT_TEXTURE0, proj, true, true));
    groups["ufo0"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(LIGHT_TEXTURE1, proj, true, true));
    groups["ufo1"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(LIGHT_TEXTURE2, proj, true, true));
    groups["ufo2"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(LIGHT_TEXTURE3, proj, true, true));
    groups["ufo3"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(vector<float,4>(1.0f,0.65f,0.2f,1.0f), proj, true, true));
    groups["ufo_light_0"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(vector<float,4>(0.9f,0.85f,0.95f,1.0f), proj, true, false));
    groups["ufo_light_1"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(vector<float,4>(0.9f,0.85f,0.75f,1.0f), proj, true, true));
    lightSources.back()->setLightColor(vector<float,4>(0.9f,0.85f,0.75f,1.0f));
    groups["ufo_light_2"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(vector<float,4>(0.9f,0.3f,0.4f,1.0f), proj, true, true));
    groups["ufo_light_3"]->addObject(lightSources.back().get());

    lightSources.push_back(std::make_shared<spotLight>(vector<float,4>(0.2f,0.5f,0.95f,1.0f), proj, true, true));
    groups["ufo_light_4"]->addObject(lightSources.back().get());

    for(auto& source: lightSources){
        for(auto& [_,graph]: graphics){
            graph->bind(source.get());
        }
        source->setLightDropFactor(0.05f);
    }
}

void testScene::mouseEvent(float frameTime)
{
    float sensitivity = mouse->sensitivity * frameTime;

    for(uint32_t i=0; i < resourceCount; i++){
        graphics["base"]->readStorageBuffer(i, primitiveNumber, farBlurDepth);
        if(primitiveNumber != std::numeric_limits<uint32_t>::max()){
            break;
        }
    }

    glfwSetScrollCallback(window,[](GLFWwindow*, double, double) {});

    if(double x = 0, y = 0; mouse->pressed(GLFW_MOUSE_BUTTON_LEFT)){
        glfwGetCursorPos(window,&x,&y);
        cameras["base"]->rotateX(sensitivity * static_cast<float>(mousePos[1] - y), {1.0f,0.0f,0.0f});
        cameras["base"]->rotateY(sensitivity * static_cast<float>(mousePos[0] - x), {0.0f,0.0f,1.0f});
        mousePos = {x,y};

        auto scale = [](double pos, double ex) -> float { return static_cast<float>(pos / ex);};
        for(uint32_t i=0; i < resourceCount; i++){
            graphics["base"]->updateStorageBuffer(i, scale(mousePos[0], extent[0]), scale(mousePos[1], extent[1]));
        }
    } else {
        glfwGetCursorPos(window,&mousePos[0],&mousePos[1]);
    }

    if(mouse->released(GLFW_MOUSE_BUTTON_LEFT)){
        for(auto& [key, object]: objects){
            if(object->comparePrimitive(primitiveNumber)){
                if(controledObject){
                    controledObject->setOutlining(false);
                }
                controledObject = object.get();
                controledObjectName = key;
                controledObject->setOutlining(true && controledObjectEnableOutlighting, 0.03f,
                    {
                        controledObjectOutlightingColor[0],
                        controledObjectOutlightingColor[1],
                        controledObjectOutlightingColor[2],
                        controledObjectOutlightingColor[3]
                    }
                );

                for(auto& [_,graph]: graphics){
                    graph->updateCmdFlags();
                }
            }
        }
    }

#ifdef SECOND_VIEW_WINDOW
    if(mouse->released(GLFW_MOUSE_BUTTON_RIGHT))
    {
        if(auto q = convert(inverse(cameras["base"]->getViewMatrix())); cameras.count("view") > 0){
            cameras["view"]->setTranslation(q.translation().im()).setRotation(q.rotation());
        }
    }
#endif
}

void testScene::keyboardEvent(float frameTime)
{
    float sensitivity = 8.0f * frameTime;

    if(!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_A)) cameras["base"]->translate(-sensitivity*cameras["base"]->getViewMatrix()[0].dvec());
    if(!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_X)) cameras["base"]->translate(-sensitivity*cameras["base"]->getViewMatrix()[1].dvec());
    if(!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_W)) cameras["base"]->translate(-sensitivity*cameras["base"]->getViewMatrix()[2].dvec());
    if(!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_D)) cameras["base"]->translate( sensitivity*cameras["base"]->getViewMatrix()[0].dvec());
    if(!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_Z)) cameras["base"]->translate( sensitivity*cameras["base"]->getViewMatrix()[1].dvec());
    if(!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_S)) cameras["base"]->translate( sensitivity*cameras["base"]->getViewMatrix()[2].dvec());

    auto rotateControled = [this](const float& ang, const vector<float,3>& ax){
        if(bool foundInGroups = false; this->controledObject){
            for(auto& [_,group]: this->groups){
                if(group->findObject(this->controledObject)){
                    group->rotate(ang,ax);
                    foundInGroups = true;
                }
            }
            if(!foundInGroups) this->controledObject->rotate(ang,ax);
        }
    };

    if(board->pressed(GLFW_KEY_KP_4)) rotateControled(radians(0.5f),{0.0f,0.0f,1.0f});
    if(board->pressed(GLFW_KEY_KP_6)) rotateControled(radians(-0.5f),{0.0f,0.0f,1.0f});
    if(board->pressed(GLFW_KEY_KP_8)) rotateControled(radians(0.5f),{1.0f,0.0f,0.0f});
    if(board->pressed(GLFW_KEY_KP_5)) rotateControled(radians(-0.5f),{1.0f,0.0f,0.0f});
    if(board->pressed(GLFW_KEY_KP_7)) rotateControled(radians(0.5f),{0.0f,1.0f,0.0f});
    if(board->pressed(GLFW_KEY_KP_9)) rotateControled(radians(-0.5f),{0.0f,1.0f,0.0f});

    auto translateControled = [this](const vector<float,3>& tr){
        if(bool foundInGroups = false; this->controledObject){
            for(auto& [_,group]: this->groups){
                if(group->findObject(this->controledObject)){
                    group->translate(tr);
                    foundInGroups = true;
                }
            }
            if(!foundInGroups) this->controledObject->translate(tr);
        }
    };

    if(board->pressed(GLFW_KEY_LEFT))           translateControled(sensitivity*vector<float,3>(-1.0f, 0.0f, 0.0f));
    if(board->pressed(GLFW_KEY_RIGHT))          translateControled(sensitivity*vector<float,3>( 1.0f, 0.0f, 0.0f));
    if(board->pressed(GLFW_KEY_UP))             translateControled(sensitivity*vector<float,3>( 0.0f, 1.0f, 0.0f));
    if(board->pressed(GLFW_KEY_DOWN))           translateControled(sensitivity*vector<float,3>( 0.0f,-1.0f, 0.0f));
    if(board->pressed(GLFW_KEY_KP_ADD))         translateControled(sensitivity*vector<float,3>( 0.0f, 0.0f, 1.0f));
    if(board->pressed(GLFW_KEY_KP_SUBTRACT))    translateControled(sensitivity*vector<float,3>( 0.0f, 0.0f,-1.0f));

    if(board->released(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window,GLFW_TRUE);

    if(board->released(GLFW_KEY_T)) {
        objects["bee0"]->changeAnimation(objects["bee0"]->getAnimationIndex() == 0 ? 1 : 0, 0.5f);
    }

    if(board->released(GLFW_KEY_N)) {
        std::random_device device;
        std::uniform_real_distribution dist(0.3f, 1.0f);
        vector<float,4> newColor = vector<float,4>(dist(device), dist(device), dist(device), 1.0f);

        lightSources.push_back(std::make_shared<spotLight>(newColor, perspective(radians(90.0f), 1.0f, 0.1f, 20.0f), true, true));
        lightSources.back()->setLightDropFactor(0.2f);

        objects["ufo" + std::to_string(ufoCounter)] = std::make_shared<baseObject>(models["ufo"].get());
        objects["ufo" + std::to_string(ufoCounter)]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});

        groups["ufo0" + std::to_string(ufoCounter)] = std::make_shared<group>();
        groups["ufo0" + std::to_string(ufoCounter)]->translate(cameras["base"]->getTranslation());
        groups["ufo0" + std::to_string(ufoCounter)]->addObject(lightSources.back().get());
        groups["ufo0" + std::to_string(ufoCounter)]->addObject(objects["ufo" + std::to_string(ufoCounter)].get());

        for(auto& [_,graph]: graphics){
            graph->bind(lightSources.back().get());
            graph->bind(objects["ufo" + std::to_string(ufoCounter++)].get());
        }
    }

    if(board->released(GLFW_KEY_B)) {
        if(ufoCounter > 4 && app->deviceWaitIdle() == VK_SUCCESS) {
            for(auto& [_,graph]: graphics){
                graph->remove(objects["ufo" + std::to_string(ufoCounter - 1)].get());
                graph->remove(lightSources.back().get());
            }
            lightSources.pop_back();
            objects.erase("ufo" + std::to_string(ufoCounter--));
        }
    }
}

void testScene::updates(float frameTime)
{
    globalTime += frameTime;

    skyboxObjects["stars"]->rotate(0.1f * frameTime, normalize(vector<float,3>(1.0f,1.0f,1.0f)));
    objects["helmet"]->
        rotate(0.5f * frameTime, normalize(vector<float,3>(0.0f,0.0f,1.0f))).
        translate(vector<float,3>(0.0f, 0.0f, 0.005f * std::sin(0.5f * globalTime)));

    graphics["base"]->setBlitFactor(blitFactor).setBlurDepth(1.02f * farBlurDepth);
}

