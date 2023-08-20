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

testPos::testPos(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath):
    ExternalPath(ExternalPath),
    window(window),
    app(app)
{}

void testPos::resize(uint32_t WIDTH, uint32_t HEIGHT)
{
    this->WIDTH = WIDTH;
    this->HEIGHT = HEIGHT;

    globalCamera->recreate(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    localCamera->recreate(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    globalSpaceView->setExtentAndOffset({static_cast<uint32_t>(WIDTH), static_cast<uint32_t>(HEIGHT)});
    localView->setExtentAndOffset({static_cast<uint32_t>(WIDTH / 3), static_cast<uint32_t>(HEIGHT / 3)}, {static_cast<int32_t>(WIDTH / 2), static_cast<int32_t>(HEIGHT / 2)});

    auto createGraphics = [this](deferredGraphics* graph){
        graph->destroyGraphics();
        graph->createGraphics(window, app->getSurface());
    };

    createGraphics(globalSpaceView);
    createGraphics(localView);
}

void testPos::create(uint32_t WIDTH, uint32_t HEIGHT)
{
    this->WIDTH = WIDTH;
    this->HEIGHT = HEIGHT;

    globalCamera = new baseCamera(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);
    localCamera = new baseCamera(45.0f, (float) WIDTH / (float) HEIGHT, 0.1f);

    globalSpaceView = new deferredGraphics{ExternalPath / "core/deferredGraphics/spv", {WIDTH, HEIGHT}};
    localView = new deferredGraphics{ExternalPath / "core/deferredGraphics/spv", {WIDTH/3, HEIGHT/3}, {static_cast<int32_t>(WIDTH / 2), static_cast<int32_t>(HEIGHT / 2)}};

    auto createGraphics = [this](deferredGraphics* graph, camera* camera){
        app->setGraphics(graph);
        graph->createCommandPool();
        graph->createEmptyTexture();
        graph->bindCameraObject(camera, true);
        graph->createGraphics(window, app->getSurface());
        graph->setMinAmbientFactor(minAmbientFactor);
    };

    createGraphics(globalSpaceView, globalCamera);
    createGraphics(localView, localCamera);

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
    auto destroyObjects = [this](deferredGraphics* graph){
        for (auto& [_,model]: models)           graph->destroyModel(model);
        for (auto& [_,object]: cameraObject3D)  graph->removeObject(object);
        for (auto& [_,object]: staticObject3D)  graph->removeObject(object);
        for (auto& lightSource: lightSources)   graph->removeLightSource(lightSource);
        for (auto& object: skyboxObjects)       graph->removeObject(object);
    };

    destroyObjects(globalSpaceView);
    destroyObjects(localView);

    for(auto& lightPoint: lightPoints) delete lightPoint;
    lightPoints.clear();
    for(auto& object: skyboxObjects) delete object;
    skyboxObjects.clear();
    for(auto& [_,object]: staticObject3D) delete object;
    staticObject3D.clear();
    for(auto& [_, model]: models) delete model;
    models.clear();
    for(auto& [view, object]: cameraObject3D){
        delete view; delete object;
    }
    cameraObject3D.clear();

    auto destroyGraphics = [](deferredGraphics* graph, camera* camera){
        graph->destroyGraphics();
        graph->destroyEmptyTextures();
        graph->destroyCommandPool();
        graph->removeCameraObject(camera);
        delete graph;
    };
    destroyGraphics(globalSpaceView, globalCamera);
    destroyGraphics(localView, localCamera);
}

void testPos::loadModels()
{
    models["cameraModel"] = new class plyModel(ExternalPath / "dependences/model/pyramid.ply");
    models["ojectModel"] = new class plyModel(ExternalPath / "dependences/model/plytest.ply");
    models["cubeModel"] = new class plyModel(ExternalPath / "dependences/model/cube.ply");

    for(auto& [_, model]: models){
        globalSpaceView->createModel(model);
    }
}

void testPos::createLight()
{
    lightPoints.push_back(new isotropicLight(lightSources, 10.0f));
    lightPoints.back()->setLightColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    lightPoints.back()->setLightDropFactor(1.0f);

    for(auto& source: lightSources){
        globalSpaceView->bindLightSource(source, true);
    }

    matrix<float,4,4> proj = perspective(radians(90.0f), 1.0f, 0.01f, 10.0f);
    dualQuaternion<float> Q = convert(*cameraObject3D.begin()->first);

    lightSources.push_back(new spotLight(proj));
    lightSources.back()->setLightColor({1.0f,1.0f,1.0f,1.0f});
    lightSources.back()->setLightDropFactor(1.0f);
    lightSources.back()->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector()/maximum(maxSize));
    globalSpaceView->bindLightSource(lightSources.back(), true);

    lightSources.push_back(new spotLight(proj));
    lightSources.back()->setLightColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    lightSources.back()->setLightDropFactor(1.0f);
    lightSources.back()->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector()/maximum(maxSize));
    localView->bindLightSource(lightSources.back(), true);
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

    skyboxObjects.push_back(new skyboxObject(SKYBOX));
    skyboxObjects.back()->scale(vector<float,3>(20000.0f,20000.0f,20000.0f));
    skyboxObjects.back()->getTexture()->setMipLevel(0.85f);

    for(auto& object: skyboxObjects){
        globalSpaceView->bindObject(object, true);
        localView->bindObject(object, false);
    }
    skyboxObjects.back()->setColorFactor(vector<float,4>(0.25));

    for (auto &entry : std::filesystem::directory_iterator(ExternalPath / "dependences/texture_HD")){
        if(std::ifstream file(entry.path()); entry.path().extension() == ".xf"){
            matrix<float,4,4>* view = new matrix<float,4,4>(file);

            cameraObject3D[view] = new baseObject(models.at("cameraModel"));
            cameraObject3D[view]->rotate(radians(180.0f),{1.0f,0.0f,0.0f});
            cameraObject3D[view]->setConstantColor(vector<float,4>(0.9f,0.3f,0.3f,-0.2f));
            cameraObject3D[view]->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,0.0f));
            maxSize = maxAbs(maxSize, vector<float,4>(*view * vector<float,4>(0.0f,0.0f,0.0f,1.0f)).dvec());
        }
    }

    staticObject3D["object"] = new baseObject(models.at("ojectModel"));
    staticObject3D["object"]->setConstantColor(vector<float,4>(0.2f,0.8f,1.0f,1.0f));
    staticObject3D["object"]->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,1.0f));

    vector<float,3> maxObjectSize = static_cast<plyModel*>(staticObject3D["object"]->getModel())->getMaxSize();
    maxSize = maxAbs(maxObjectSize, maxSize);

    staticObject3D["cube"] = new baseObject(models.at("cubeModel"));
    staticObject3D["cube"]->setConstantColor(vector<float,4>(0.7f,0.7f,0.7f,1.0f));

    selectedObject = cameraObject3D.begin()->second;
    selectedObject->setOutlining(true, 1.5f / maximum(maxSize), {0.8f, 0.6f, 0.1f, 1.0f});
    for(auto& [view,object]: cameraObject3D){
        auto curview = *view;
        curview[0][3] /= maximum(maxSize);
        curview[1][3] /= maximum(maxSize);
        curview[2][3] /= maximum(maxSize);
        globalSpaceView->bindObject(object, true);
        object->setGlobalTransform(curview).scale(50.0f/maximum(maxSize));
    }
    for(auto& [_,object]: staticObject3D){
        globalSpaceView->bindObject(object, true);
        localView->bindObject(object, false);
        object->scale(1.0f/maximum(maxSize));
    }

    staticObject3D["cube"]->scale(2.0f);
    dualQuaternion<float> Q = convert(*cameraObject3D.begin()->first);
    localCamera->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector()/maximum(maxSize));
}

void testPos::mouseEvent(float)
{
    int primitiveNumber = INT_FAST32_MAX;
    for(uint32_t i=0; i < app->getImageCount(); i++){
        primitiveNumber = globalSpaceView->readStorageBuffer(i);
        if(primitiveNumber!=INT_FAST32_MAX)
            break;
    }

    glfwSetScrollCallback(window,[](GLFWwindow*, double, double) {
    });

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        float sensitivity = 0.001;

        double x, y;
        glfwGetCursorPos(window,&x,&y);
        globalCamera->rotateX(sensitivity * static_cast<float>(yMpos - y), {1.0f,0.0f,0.0f});
        globalCamera->rotateY(sensitivity * static_cast<float>(xMpos - x), {0.0f,0.0f,1.0f});
        xMpos = x;
        yMpos = y;

        for(uint32_t i=0; i < app->getImageCount(); i++){
            globalSpaceView->updateStorageBuffer(i, 2.0f * static_cast<float>(xMpos)/WIDTH - 1.0f , 2.0f * static_cast<float>(yMpos)/HEIGHT - 1.0f);
        }
    }
    else if(mouse1Stage == GLFW_PRESS && glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == 0)
    {
        for(auto& [view, object]: cameraObject3D){
            if(object->comparePrimitive(primitiveNumber)){
                selectedObject->setOutlining(false);
                auto curview = *view;
                curview[0][3] /= maximum(maxSize);
                curview[1][3] /= maximum(maxSize);
                curview[2][3] /= maximum(maxSize);
                dualQuaternion<float> Q = convert(curview);

                localCamera->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector());

                (*(lightSources.rbegin()+0))->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector());
                (*(lightSources.rbegin()+1))->setRotation(radians(180.0f),{1.0f,0.0f,0.0f}).rotate(Q.rotation()).setTranslation(Q.translation().vector());

                object->setOutlining(true, 1.5f / maximum(maxSize), {0.8f, 0.6f, 0.1f, 1.0f});
                selectedObject = object;

                globalSpaceView->updateCmdFlags();
                localView->updateCmdFlags();
            }
        }
    }
    else
    {
        glfwGetCursorPos(window,&xMpos,&yMpos);
    }
    mouse1Stage = glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT);
}

void testPos::keyboardEvent(float frameTime)
{
    float sensitivity = 1.0f*frameTime;
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        lightPoints.back()->translate(-sensitivity*globalCamera->getViewMatrix()[2].dvec());
        globalCamera->translate(-sensitivity*globalCamera->getViewMatrix()[2].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        lightPoints.back()->translate(sensitivity*globalCamera->getViewMatrix()[2].dvec());
        globalCamera->translate(sensitivity*globalCamera->getViewMatrix()[2].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        lightPoints.back()->translate(-sensitivity*globalCamera->getViewMatrix()[0].dvec());
        globalCamera->translate(-sensitivity*globalCamera->getViewMatrix()[0].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
    {
        lightPoints.back()->translate(sensitivity*globalCamera->getViewMatrix()[0].dvec());
        globalCamera->translate(sensitivity*globalCamera->getViewMatrix()[0].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_Z) == GLFW_PRESS)
    {
        lightPoints.back()->translate(sensitivity*globalCamera->getViewMatrix()[1].dvec());
        globalCamera->translate(sensitivity*globalCamera->getViewMatrix()[1].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_X) == GLFW_PRESS)
    {
        lightPoints.back()->translate(-sensitivity*globalCamera->getViewMatrix()[1].dvec());
        globalCamera->translate(-sensitivity*globalCamera->getViewMatrix()[1].dvec());
    }

    if(glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window,GLFW_TRUE);
    }

    if(glfwGetKey(window,GLFW_KEY_KP_0) == GLFW_PRESS)
    {
        minAmbientFactor -= minAmbientFactor > 0.011 ? 0.01f : 0.0f;
        globalSpaceView->setMinAmbientFactor(minAmbientFactor);
        localView->setMinAmbientFactor(minAmbientFactor);
    }
    if(glfwGetKey(window,GLFW_KEY_KP_2) == GLFW_PRESS)
    {
        minAmbientFactor += 0.01f;
        globalSpaceView->setMinAmbientFactor(minAmbientFactor);
        localView->setMinAmbientFactor(minAmbientFactor);
    }
}

void testPos::updates(float frameTime)
{
    globalTime += frameTime;
}

