#include "testScene.h"
#include "deferredGraphics.h"
#include "graphicsManager.h"
#include "imguiGraphics.h"
#include "gltfmodel.h"
#include "spotLight.h"
#include "baseObject.h"
#include "group.h"
#include "baseCamera.h"
#include "dualQuaternion.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <random>

testScene::testScene(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath):
    ExternalPath(ExternalPath),
    window(window),
    app(app),
    mouse(new controller(window, glfwGetMouseButton)),
    board(new controller(window, glfwGetKey))
{}

testScene::~testScene(){
    delete mouse;
    delete board;
}

void testScene::resize(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};

    cameras["base"]->recreate(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["base"]->setExtentAndOffset({static_cast<uint32_t>(WIDTH), static_cast<uint32_t>(HEIGHT)});

    cameras["view"]->recreate(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["view"]->setExtentAndOffset({static_cast<uint32_t>(WIDTH / 3), static_cast<uint32_t>(HEIGHT / 3)}, {static_cast<int32_t>(WIDTH / 2), static_cast<int32_t>(HEIGHT / 2)});

    for(auto& [_,graph]: graphics){
        graph->destroyGraphics();
        graph->createGraphics();
        graph->updateDescriptorSets();
    }

    gui->destroyGraphics();
    gui->createGraphics();
}

void testScene::create(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};

    cameras["base"] = new baseCamera(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["base"] = new deferredGraphics{ExternalPath / "core/deferredGraphics/spv", {WIDTH, HEIGHT}};
    app->setGraphics(graphics["base"]);
    graphics["base"]->bind(cameras["base"]);
    graphics["base"]->setEnable("TransparentLayer", true).setEnable("Skybox", true).setEnable("Blur", true).setEnable("Bloom", true).setEnable("SSAO", true).setEnable("SSLR", true).setEnable("Scattering", true).setEnable("Shadow", true);

    cameras["view"] = new baseCamera(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["view"] = new deferredGraphics{ExternalPath / "core/deferredGraphics/spv", {WIDTH/3, HEIGHT/3}, {static_cast<int32_t>(WIDTH / 2), static_cast<int32_t>(HEIGHT / 2)}};
    app->setGraphics(graphics["view"]);
    graphics["view"]->bind(cameras["view"]);
    graphics["view"]->setEnable("TransparentLayer", true).setEnable("Skybox", true).setEnable("Blur", true).setEnable("Bloom", true).setEnable("SSAO", true).setEnable("SSLR", true).setEnable("Scattering", true).setEnable("Shadow", true);

    gui = new imguiGraphics;
    gui->setInstance(app->getInstance());
    app->setGraphics(gui);

    for(auto& [_,graph]: graphics){
        graph->createGraphics();
        graph->updateDescriptorSets();
    }

    gui->createGraphics();

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

    ImGuiIO io = ImGui::GetIO();
    if(!io.WantCaptureMouse)    mouseEvent(frameTime);
    if(!io.WantCaptureKeyboard) keyboardEvent(frameTime);

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()
    bool showDemoWindow = true;
    ImGui::ShowDemoWindow(&showDemoWindow);

    updates(frameTime);

    for(auto& [_,object]: objects){
        object->animationTimer += timeScale * frameTime;
        object->updateAnimation(frameNumber);
    }
}

void testScene::destroy()
{
    for(auto& [_,graph]: graphics){
        for (auto& [_,object]: objects)            graph->remove(object);
        for (auto& [_,object]: staticObjects)      graph->remove(object);
        for (auto& [_,object]: skyboxObjects)      graph->remove(object);
        for (auto& [_,model]: models)              graph->destroy(model);
        for (auto& light: lightSources)            graph->remove(light);
    }
    for(auto& [_,lightPoint]: lightPoints)  delete lightPoint;
    for(auto& [_,object]: objects)          delete object;
    for(auto& [_,object]: staticObjects)    delete object;
    for(auto& [_,object]: skyboxObjects)    delete object;
    for(auto& [_,model]: models)            delete model;
    for(auto& light: lightSources)          delete light;

    lightPoints.clear();
    objects.clear();
    objects.clear();
    staticObjects.clear();
    models.clear();
    lightSources.clear();
    skyboxObjects.clear();

    for(auto& [key,graph]: graphics){
        graph->destroyGraphics();
        graph->remove(cameras[key]);
        delete graph;
    }

    gui->destroyGraphics();
}

void testScene::loadModels()
{
    models["bee"] = new class gltfModel(ExternalPath / "dependences/model/glb/Bee.glb", 6);
    models["box"] = new class gltfModel(ExternalPath / "dependences/model/glb/Box.glb");
    models["sponza"] = new class gltfModel(ExternalPath / "dependences/model/glTF/Sponza/Sponza.gltf");
    models["duck"] = new class gltfModel(ExternalPath / "dependences/model/glb/Duck.glb");
    models["ufo"] = new class gltfModel(ExternalPath / "dependences/model/glb/RetroUFO.glb");

    for(auto& [_,model]: models){
        graphics["base"]->create(model);
    }
}

void testScene::createObjects()
{
    staticObjects["sponza"] = new baseObject(models["sponza"]);
    staticObjects["sponza"]->rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({3.0f,3.0f,3.0f});

    objects["bee0"] = new baseObject(models["bee"], 0, 3);
    objects["bee0"]->translate({3.0f,0.0f,0.0f}).rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({0.2f,0.2f,0.2f});

    objects["bee1"] = new baseObject(models["bee"], 3, 3);
    objects["bee1"]->translate({-3.0f,0.0f,0.0f}).rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({0.2f,0.2f,0.2f});
    objects["bee1"]->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.7f));
    objects["bee1"]->animationTimer = 1.0f;
    objects["bee1"]->animationIndex = 1;

    objects["duck"] = new baseObject(models["duck"]);
    objects["duck"]->translate({0.0f,0.0f,3.0f}).rotate(radians(90.0f),{1.0f,0.0f,0.0f}).scale({3.0f});
    objects["duck"]->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.8f));

    objects["lightBox"] = new baseObject(models["box"]);
    objects["lightBox"]->setBloomColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    groups["lightBox"] = new group;
    groups["lightBox"]->addObject(objects["lightBox"]);

    for(auto key = "ufo" + std::to_string(ufoCounter); ufoCounter < 4; ufoCounter++, key = "ufo" + std::to_string(ufoCounter)){
        objects[key] = new baseObject(models["ufo"]);
        objects[key]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
        objects[key]->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.8f));
        groups[key] = new group;
        groups[key]->addObject(objects["ufo" + std::to_string(ufoCounter)]);
    }

    skyboxObjects["lake"] = new skyboxObject({
        ExternalPath / "dependences/texture/skybox/left.jpg",
        ExternalPath / "dependences/texture/skybox/right.jpg",
        ExternalPath / "dependences/texture/skybox/front.jpg",
        ExternalPath / "dependences/texture/skybox/back.jpg",
        ExternalPath / "dependences/texture/skybox/top.jpg",
        ExternalPath / "dependences/texture/skybox/bottom.jpg"
    });
    // skyboxObjects["lake"]->setColorFactor(vector<float,4>(0.5));
    skyboxObjects["lake"]->scale({200.0f,200.0f,200.0f});

    skyboxObjects["stars"] = new skyboxObject({
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
            graph->bind(object);
        }
        for(auto& [_,object]: staticObjects){
            graph->bind(object);
        }
        for(auto& [_, object]: skyboxObjects){
            graph->bind(object);
        }
    }
}

void testScene::createLight()
{
    std::filesystem::path LIGHT_TEXTURE0  = ExternalPath / "dependences/texture/icon.PNG";
    std::filesystem::path LIGHT_TEXTURE1  = ExternalPath / "dependences/texture/light1.jpg";
    std::filesystem::path LIGHT_TEXTURE2  = ExternalPath / "dependences/texture/light2.jpg";
    std::filesystem::path LIGHT_TEXTURE3  = ExternalPath / "dependences/texture/light3.jpg";

    lightPoints["lightBox"] = new isotropicLight(lightSources);
    lightPoints["lightBox"]->setLightColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    groups["lightBox"]->addObject(lightPoints["lightBox"]);

    matrix<float,4,4> proj = perspective(radians(90.0f), 1.0f, 0.1f, 20.0f);

    lightSources.push_back(new spotLight(LIGHT_TEXTURE0, proj, true, true));
    groups["ufo0"]->addObject(lightSources.back());

    lightSources.push_back(new spotLight(LIGHT_TEXTURE1, proj, true, true));
    groups["ufo1"]->addObject(lightSources.back());

    lightSources.push_back(new spotLight(LIGHT_TEXTURE2, proj, true, true));
    groups["ufo2"]->addObject(lightSources.back());

    lightSources.push_back(new spotLight(LIGHT_TEXTURE3, proj, true, true));
    groups["ufo3"]->addObject(lightSources.back());

    for(auto& source: lightSources){
        for(auto& [_,graph]: graphics){
            graph->bind(source);
        }
        source->setLightDropFactor(0.2f);
    }
}

void testScene::mouseEvent(float frameTime)
{
    float sensitivity = mouse->sensitivity * frameTime;
    uint32_t imageCount = app->getSwapChain()->getImageCount();

    int primitiveNumber = INT_FAST32_MAX;
    for(uint32_t i=0; i < imageCount; i++){
        if(primitiveNumber = graphics["base"]->readStorageBuffer(i); primitiveNumber != INT_FAST32_MAX)
            break;
    }

    glfwSetScrollCallback(window,[](GLFWwindow*, double, double) {});

    if(double x = 0, y = 0; mouse->pressed(GLFW_MOUSE_BUTTON_LEFT)){
        glfwGetCursorPos(window,&x,&y);
        cameras["base"]->rotateX(sensitivity * static_cast<float>(mousePos[1] - y), {1.0f,0.0f,0.0f});
        cameras["base"]->rotateY(sensitivity * static_cast<float>(mousePos[0] - x), {0.0f,0.0f,1.0f});
        mousePos = {x,y};

        auto scale = [](double pos, double ex) -> float { return static_cast<float>(2.0 * pos / ex - 1.0);};
        for(uint32_t i=0; i < imageCount; i++){
            graphics["base"]->updateStorageBuffer(i, scale(mousePos[0],extent[0]), scale(mousePos[1],extent[1]));
        }
    } else {
        glfwGetCursorPos(window,&mousePos[0],&mousePos[1]);
    }

    if(mouse->released(GLFW_MOUSE_BUTTON_LEFT)){
        for(auto& [key, object]: objects){
            object->setOutlining(false);
            if(object->comparePrimitive(primitiveNumber)){
                std::random_device device;
                std::uniform_real_distribution dist(0.3f, 1.0f);

                object->setOutlining(true, 0.03f, {dist(device), dist(device), dist(device), dist(device)});
                controledObject = object;
                std::cout<< key <<std::endl;

                for(auto& [_,graph]: graphics){
                    graph->updateCmdFlags();
                }
            }
        }
    }

    if(mouse->released(GLFW_MOUSE_BUTTON_RIGHT))
    {
        if(auto q = convert(inverse(cameras["base"]->getViewMatrix())); cameras.count("view") > 0){
            cameras["view"]->setTranslation(q.translation().vector()).setRotation(q.rotation());
        }
    }
}

void testScene::keyboardEvent(float frameTime)
{
    float sensitivity = 5.0f*frameTime;

    if(board->pressed(GLFW_KEY_A)) cameras["base"]->translate(-sensitivity*cameras["base"]->getViewMatrix()[0].dvec());
    if(board->pressed(GLFW_KEY_X)) cameras["base"]->translate(-sensitivity*cameras["base"]->getViewMatrix()[1].dvec());
    if(board->pressed(GLFW_KEY_W)) cameras["base"]->translate(-sensitivity*cameras["base"]->getViewMatrix()[2].dvec());
    if(board->pressed(GLFW_KEY_D)) cameras["base"]->translate( sensitivity*cameras["base"]->getViewMatrix()[0].dvec());
    if(board->pressed(GLFW_KEY_Z)) cameras["base"]->translate( sensitivity*cameras["base"]->getViewMatrix()[1].dvec());
    if(board->pressed(GLFW_KEY_S)) cameras["base"]->translate( sensitivity*cameras["base"]->getViewMatrix()[2].dvec());

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

    if(board->released(GLFW_KEY_O)) {
        for(auto& [_,object]: objects){
            if(controledObject == object){
                std::random_device device;
                std::uniform_real_distribution dist(0.3f, 1.0f);
                object->setOutlining(!object->getOutliningEnable(), 0.03f, {dist(device), dist(device), dist(device), dist(device)});
            }
        }
        for(auto& [_,graph]: graphics){
            graph->updateCmdFlags();
        }
    }

    if(board->released(GLFW_KEY_T)) {
        objects["bee0"]->changeAnimationFlag = true;
        objects["bee0"]->startTimer = objects["bee0"]->animationTimer;
        objects["bee0"]->changeAnimationTime = 0.5f;
        if(objects["bee0"]->animationIndex == 0){
            objects["bee0"]->newAnimationIndex = 1;
        } else if(objects["bee0"]->animationIndex == 1){
            objects["bee0"]->newAnimationIndex = 0;
        }
    }

    if(board->released(GLFW_KEY_N)) {
        std::random_device device;
        std::uniform_real_distribution dist(0.3f, 1.0f);

        lightSources.push_back(new spotLight(perspective(radians(90.0f), 1.0f, 0.1f, 20.0f), true, true));
        lightSources.back()->setLightColor({dist(device), dist(device), dist(device), 1.0f});
        lightSources.back()->setLightDropFactor(0.2f);

        objects["ufo" + std::to_string(ufoCounter)] = new baseObject(models["ufo"]);
        objects["ufo" + std::to_string(ufoCounter)]->rotate(radians(90.0f),{1.0f,0.0f,0.0f});

        groups["ufo0" + std::to_string(ufoCounter)] = new group;
        groups["ufo0" + std::to_string(ufoCounter)]->translate(cameras["base"]->getTranslation());
        groups["ufo0" + std::to_string(ufoCounter)]->addObject(lightSources.back());
        groups["ufo0" + std::to_string(ufoCounter)]->addObject(objects["ufo" + std::to_string(ufoCounter)]);

        for(auto& [_,graph]: graphics){
            graph->bind(lightSources.back());
            graph->bind(objects["ufo" + std::to_string(ufoCounter)]);
        }
        ufoCounter++;
    }

    if(board->released(GLFW_KEY_B)) {
        app->deviceWaitIdle();
        if(ufoCounter > 4) {
            for(auto& [_,graph]: graphics){
                graph->remove(objects["ufo" + std::to_string(ufoCounter - 1)]);
                graph->remove(lightSources.back());
            }
            lightSources.pop_back();
            objects.erase("ufo" + std::to_string(ufoCounter));
            ufoCounter--;
        }
    }

    if(board->pressed(GLFW_KEY_KP_0)) {
        minAmbientFactor -= minAmbientFactor > 0.011 ? 0.01f : 0.0f;
        for(auto& [_,graph]: graphics){
            graph->setMinAmbientFactor(minAmbientFactor);
        }
    }
    if(board->pressed(GLFW_KEY_KP_2)) {
        minAmbientFactor += 0.01f;
        for(auto& [_,graph]: graphics){
            graph->setMinAmbientFactor(minAmbientFactor);
        }
    }

    if(board->released(GLFW_KEY_G)){
        enableScatteringRefraction = !enableScatteringRefraction;
        for(auto& [_,graph]: graphics){
            graph->setScatteringRefraction(enableScatteringRefraction);
        }
    }

    if(board->pressed(GLFW_KEY_LEFT_BRACKET))   timeScale -= timeScale > 0.0051f ? 0.005f : 0.0f;
    if(board->pressed(GLFW_KEY_RIGHT_BRACKET))  timeScale += 0.005f;
}

void testScene::updates(float frameTime)
{
    globalTime += frameTime;

    skyboxObjects["stars"]->rotate(0.1f*frameTime,normalize(vector<float,3>(1.0f,1.0f,1.0f)));
}

