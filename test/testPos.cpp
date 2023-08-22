#include "testPos.h"

#include "deferredGraphics.h"
#include "graphicsManager.h"
#include "spotLight.h"
#include "baseObject.h"
#include "baseCamera.h"
#include "plymodel.h"
#include "dualQuaternion.h"
#include "matrix.h"

#include <glfw3.h>
#include <fstream>

namespace {
VkExtent2D getSmallWindowExent(uint32_t WIDTH, uint32_t HEIGHT, float aspect){
    uint32_t HEIGHT_2 = std::min(WIDTH / 2, HEIGHT / 2);
    uint32_t WIDTH_2 = static_cast<uint32_t>(aspect * (float)HEIGHT_2);

    if(WIDTH_2 > WIDTH / 2) {
        float a = (float)WIDTH / 2.0f / (float)WIDTH_2;
        WIDTH_2 = static_cast<uint32_t>(a * (float)WIDTH_2);
        HEIGHT_2 = static_cast<uint32_t>(a * (float)HEIGHT_2);
    }
    return {WIDTH_2, HEIGHT_2};
}

matrix<float,4,4>& correctMatrix(matrix<float,4,4>& m){
    m[1][1] *= - 1.0f;
    m[1][2] *= - 1.0f;
    m[2][2] *= - 1.0f;
    m[3][2] = - 1.0f;
    m[2][3] = - 0.01f; // - 2 * near
    return m;
}
}

testPos::testPos(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath):
    ExternalPath(ExternalPath),
    window(window),
    app(app),
    mouse(new controller(window, glfwGetMouseButton)),
    board(new controller(window, glfwGetKey))
{
    mouse->sensitivity = 0.4f;
}

void testPos::resize(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};

    matrix<float,4,4> m(0.0f);
    if(std::ifstream file(ExternalPath / "dependences/texture_HD/camera_matrix.txt"); file.is_open()){
        m = matrix<float,4,4>(file);
        cameras["view"]->setProjMatrix(correctMatrix(m));
    }

    cameras["base"]->recreate(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["base"]->setExtentAndOffset({static_cast<uint32_t>(WIDTH), static_cast<uint32_t>(HEIGHT)});
    graphics["view"]->setExtentAndOffset(getSmallWindowExent(WIDTH, HEIGHT, (- m[1][1] / m[0][0])), {static_cast<int32_t>(WIDTH / 2) - 4, static_cast<int32_t>(HEIGHT / 2) - 4});

    for(auto& [_,graph]: graphics){
        graph->destroyGraphics();
        graph->createGraphics(window, app->getSurface());
    }
}

void testPos::create(uint32_t WIDTH, uint32_t HEIGHT)
{
    extent = {WIDTH, HEIGHT};

    cameras["view"] = new baseCamera();
    matrix<float,4,4> m(0.0f);
    if(std::ifstream file(ExternalPath / "dependences/texture_HD/camera_matrix.txt"); file.is_open()){
        m = matrix<float,4,4>(file);
        cameras["view"]->setProjMatrix(correctMatrix(m));
    }

    cameras["base"] = new baseCamera(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    graphics["base"] = new deferredGraphics{ExternalPath / "core/deferredGraphics/spv", {WIDTH, HEIGHT}};
    graphics["view"] = new deferredGraphics{ExternalPath / "core/deferredGraphics/spv", getSmallWindowExent(WIDTH, HEIGHT, (- m[1][1] / m[0][0])), {static_cast<int32_t>(WIDTH / 2) - 4 , static_cast<int32_t>(HEIGHT / 2) - 4}};

    app->setGraphics(graphics["base"]);
    app->setGraphics(graphics["view"]);

    for(auto& [key,graph]: graphics){
        graph->createCommandPool();
        graph->createEmptyTexture();
        graph->bindCameraObject(cameras[key], true);
        graph->createGraphics(window, app->getSurface());
    }

    loadModels();
    createObjects();
    createLight();
}

void testPos::updateFrame(uint32_t, float frameTime)
{
    glfwPollEvents();
    mouseEvent(frameTime);
    keyboardEvent(frameTime);
    updates(frameTime);
}

void testPos::destroy()
{
    for(auto& [_,graph]: graphics){
        for (auto& [_,object]: cameraObjects)       graph->removeObject(object);
        for (auto& [_,object]: staticObjects)      graph->removeObject(object);
        for (auto& [_,object]: skyboxObjects)      graph->removeObject(object);
        for (auto& [_,model]: models)              graph->destroyModel(model);
        for (auto& [_,light]: lights)              graph->removeLightSource(light);
        for (auto& light: lightSources)            graph->removeLightSource(light);
    }
    for (auto& [_,lightPoint]: lightPoints)  delete lightPoint;
    for (auto& [view,object]: cameraObjects)  { delete object; delete view;}
    for (auto& [_,object]: staticObjects)    delete object;
    for (auto& [_,object]: skyboxObjects)    delete object;
    for (auto& [_,model]: models)            delete model;
    for (auto& [_,light]: lights)            delete light;
    for (auto& light: lightSources)          delete light;

    lightPoints.clear();
    cameraObjects.clear();
    cameraNames.clear();
    staticObjects.clear();
    skyboxObjects.clear();
    models.clear();
    lights.clear();
    lightSources.clear();

    for(auto& [key,graph]: graphics){
        graph->destroyGraphics();
        graph->destroyEmptyTextures();
        graph->destroyCommandPool();
        graph->removeCameraObject(cameras[key]);
        delete graph;
    }
}

void testPos::loadModels()
{
    models["cameraModel"] = new class plyModel(ExternalPath / "dependences/model/pyramid.ply");
    models["ojectModel"] = new class plyModel(ExternalPath / "dependences/model/plytest.ply");
    models["cubeModel"] = new class plyModel(ExternalPath / "dependences/model/cube.ply");

    for(auto& [_,model]: models){
        graphics["base"]->createModel(model);
    }
}

void testPos::createLight()
{
    lightPoints["lightBox"] = new isotropicLight(lightSources, 10.0f);
    lightPoints["lightBox"]->setLightColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    lightPoints["lightBox"]->setLightDropFactor(1.0f);

    for(auto& source: lightSources){
        graphics["base"]->bindLightSource(source, true);
    }

    matrix<float,4,4> proj = perspective(radians(90.0f), 1.0f, 0.01f, 10.0f);
    dualQuaternion<float> Q = convert(*cameraObjects.begin()->first);

    lights["base"] = new spotLight(proj, true, true);
    lights["base"]->setLightColor({1.0f,1.0f,1.0f,1.0f});
    lights["base"]->setLightDropFactor(1.0f);
    lights["base"]->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector()/maximum(maxSize));
    graphics["base"]->bindLightSource(lights["base"], true);

    lights["view"] = new spotLight(proj);
    lights["view"]->setLightColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    lights["view"]->setLightDropFactor(1.0f);
    lights["view"]->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector()/maximum(maxSize));
    graphics["view"]->bindLightSource(lights["view"], true);
}

void testPos::createObjects()
{
    std::vector<std::filesystem::path> SKYBOX = {
        ExternalPath / "dependences/texture/skybox/left.jpg",
        ExternalPath / "dependences/texture/skybox/right.jpg",
        ExternalPath / "dependences/texture/skybox/front.jpg",
        ExternalPath / "dependences/texture/skybox/back.jpg",
        ExternalPath / "dependences/texture/skybox/top.jpg",
        ExternalPath / "dependences/texture/skybox/bottom.jpg"
    };

    skyboxObjects["lake"] = new skyboxObject(SKYBOX);
    skyboxObjects["lake"]->scale(vector<float,3>(100.0f,100.0f,100.0f));
    skyboxObjects["lake"]->getTexture()->setMipLevel(0.85f);

    for(auto& [_, object]: skyboxObjects){
        graphics["base"]->bindObject(object, true);
        graphics["view"]->bindObject(object, false);
    }

    for (auto &entry : std::filesystem::directory_iterator(ExternalPath / "dependences/texture_HD")){
        if(std::ifstream file(entry.path()); entry.path().extension() == ".xf"){
            matrix<float,4,4>* view = new matrix<float,4,4>(file);

            cameraNames[view] = entry.path().string();
            cameraObjects[view] = new baseObject(models.at("cameraModel"));
            cameraObjects[view]->rotate(radians(180.0f),{1.0f,0.0f,0.0f});
            cameraObjects[view]->setConstantColor(vector<float,4>(0.9f,0.3f,0.3f,-0.2f));
            cameraObjects[view]->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,0.0f));
            maxSize = maxAbs(maxSize, vector<float,4>(*view * vector<float,4>(0.0f,0.0f,0.0f,1.0f)).dvec());
        }
    }

    staticObjects["object"] = new baseObject(models.at("ojectModel"));
    staticObjects["object"]->setConstantColor(vector<float,4>(0.2f,0.8f,1.0f,1.0f));
    staticObjects["object"]->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,1.0f));

    maxSize = maxAbs(static_cast<plyModel*>(staticObjects["object"]->getModel())->getMaxSize(), maxSize);
    staticObjects["object"]->scale(1.0f/maximum(maxSize));

    staticObjects["cube"] = new baseObject(models.at("cubeModel"));
    staticObjects["cube"]->setConstantColor(vector<float,4>(0.94f,0.94f,0.94f,1.0f));
    staticObjects["cube"]->scale(3.0f);

    selectedObject = cameraObjects.begin()->second;
    selectedObject->setOutlining(true, 1.5f / maximum(maxSize), {0.8f, 0.6f, 0.1f, 1.0f});

    for(auto [view,object]: cameraObjects){
        auto curview = *view;
        curview[0][3] /= maximum(maxSize); curview[1][3] /= maximum(maxSize); curview[2][3] /= maximum(maxSize);
        graphics["base"]->bindObject(object, true);
        object->setGlobalTransform(curview).scale(50.0f/maximum(maxSize));
    }
    for(auto& [_,object]: staticObjects){
        graphics["base"]->bindObject(object, true);
        graphics["view"]->bindObject(object, false);
    }

    dualQuaternion<float> Q = convert(*cameraObjects.begin()->first);
    cameras["view"]->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector()/maximum(maxSize));
}

void testPos::mouseEvent(float frameTime)
{
    float sensitivity = mouse->sensitivity * frameTime;

    int primitiveNumber = INT_FAST32_MAX;
    for(uint32_t i=0; i < app->getImageCount(); i++){
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
        for(uint32_t i=0; i < app->getImageCount(); i++){
            graphics["base"]->updateStorageBuffer(i, scale(mousePos[0],extent[0]), scale(mousePos[1],extent[1]));
        }
    } else {
        glfwGetCursorPos(window,&mousePos[0],&mousePos[1]);
    }

    if(mouse->released(GLFW_MOUSE_BUTTON_LEFT)){
        for(auto& [view, object]: cameraObjects){
            if(object->comparePrimitive(primitiveNumber)){
                std::cout << cameraNames[view] << std::endl;

                selectedObject->setOutlining(false);
                auto curview = *view;
                curview[0][3] /= maximum(maxSize); curview[1][3] /= maximum(maxSize); curview[2][3] /= maximum(maxSize);
                dualQuaternion<float> Q = convert(curview);

                object->setOutlining(true, 1.5f / maximum(maxSize), {0.8f, 0.6f, 0.1f, 1.0f});
                selectedObject = object;

                cameras["view"]->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector());

                graphics["base"]->updateCmdFlags();
                graphics["view"]->updateCmdFlags();

                lights["base"]->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector());
                lights["view"]->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector());
            }
        }
    }
}

void testPos::keyboardEvent(float frameTime)
{
    float sensitivity = 1.0f*frameTime;

    auto translate = [this](const vector<float,3>& tr){
        const auto& curTr = cameras["base"]->getTranslation();
        vector<float,3> resTr{0.0f};
        for(uint32_t i = 0; i < 3; i++) resTr[i] += std::abs(curTr[i] + tr[i]) < 2.5f ? tr[i] : 0.0f;
        lightPoints["lightBox"]->translate(resTr);
        cameras["base"]->translate(resTr);
    };

    if(board->pressed(GLFW_KEY_W)) translate(-sensitivity * cameras["base"]->getViewMatrix()[2].dvec());
    if(board->pressed(GLFW_KEY_S)) translate( sensitivity * cameras["base"]->getViewMatrix()[2].dvec());
    if(board->pressed(GLFW_KEY_A)) translate(-sensitivity * cameras["base"]->getViewMatrix()[0].dvec());
    if(board->pressed(GLFW_KEY_D)) translate( sensitivity * cameras["base"]->getViewMatrix()[0].dvec());
    if(board->pressed(GLFW_KEY_X)) translate(-sensitivity * cameras["base"]->getViewMatrix()[1].dvec());
    if(board->pressed(GLFW_KEY_Z)) translate( sensitivity * cameras["base"]->getViewMatrix()[1].dvec());

    if(board->released(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window,GLFW_TRUE);

    if(board->pressed(GLFW_KEY_KP_0)){
        minAmbientFactor -= minAmbientFactor > 0.011 ? 0.01f : 0.0f;
        graphics["base"]->setMinAmbientFactor(minAmbientFactor);
        graphics["view"]->setMinAmbientFactor(minAmbientFactor);
    }
    if(board->pressed(GLFW_KEY_KP_2)) {
        minAmbientFactor += 0.01f;
        graphics["base"]->setMinAmbientFactor(minAmbientFactor);
        graphics["view"]->setMinAmbientFactor(minAmbientFactor);
    }
}

void testPos::updates(float frameTime)
{
    globalTime += frameTime;
}

