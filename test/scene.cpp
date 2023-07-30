#include "scene.h"
#include "deferredGraphics.h"
#include "graphicsManager.h"
#include "gltfmodel.h"
#include "spotLight.h"
#include "baseObject.h"
#include "group.h"
#include "baseCamera.h"
#include "plymodel.h"
#include "dualQuaternion.h"

#include <glfw3.h>
#include <random>

bool updateLightCone = false;
float spotAngle = 90.0f;

scene::scene(graphicsManager *app, std::vector<deferredGraphics*> graphics, GLFWwindow* window, const std::filesystem::path& ExternalPath):
    window(window),
    app(app),
    graphics(graphics),
    ExternalPath(ExternalPath)
{}

void scene::createScene(uint32_t WIDTH, uint32_t HEIGHT, baseCamera* cameraObject)
{
    this->WIDTH = WIDTH;
    this->HEIGHT = HEIGHT;

    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);

    std::vector<std::filesystem::path> SKYBOX = {
        ExternalPath / "dependences/texture/skybox/left.jpg",
        ExternalPath / "dependences/texture/skybox/right.jpg",
        ExternalPath / "dependences/texture/skybox/front.jpg",
        ExternalPath / "dependences/texture/skybox/back.jpg",
        ExternalPath / "dependences/texture/skybox/top.jpg",
        ExternalPath / "dependences/texture/skybox/bottom.jpg"
    };

    std::vector<std::filesystem::path> SKYBOX1 = {
        ExternalPath / "dependences/texture/skybox1/left.png",
        ExternalPath / "dependences/texture/skybox1/right.png",
        ExternalPath / "dependences/texture/skybox1/front.png",
        ExternalPath / "dependences/texture/skybox1/back.png",
        ExternalPath / "dependences/texture/skybox1/top.png",
        ExternalPath / "dependences/texture/skybox1/bottom.png"
    };

    cameras = cameraObject;

    skyboxObjects.push_back(new skyboxObject(SKYBOX));
    skyboxObjects.back()->scale(vector<float,3>(200.0f,200.0f,200.0f));
    skyboxObjects.push_back(new skyboxObject(SKYBOX1));
    skyboxObjects.back()->scale(vector<float,3>(200.0f,200.0f,200.0f));

    for(size_t i = 0; i < graphics.size(); i++){
        for(auto& object: skyboxObjects){
            graphics[i]->bindObject(object, i == 0);
        }
    }
    skyboxObjects[0]->setColorFactor(vector<float,4>(0.5));

    loadModels();
    createLight();
    createObjects();

    groups.at(0)->translate(vector<float,3>(0.0f,0.0f,5.0f));
    groups.at(1)->translate(vector<float,3>(0.0f,0.0f,3.0f));
    groups.at(2)->translate(vector<float,3>(5.0f,0.0f,5.0f));
    groups.at(3)->translate(vector<float,3>(-5.0f,0.0f,5.0f));
    groups.at(4)->translate(vector<float,3>(10.0f,0.0f,5.0f));
    groups.at(5)->translate(vector<float,3>(-10.0f,0.0f,5.0f));
}

void scene::updateFrame(uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT)
{
    this->WIDTH = WIDTH;
    this->HEIGHT = HEIGHT;

    glfwPollEvents();
    mouseEvent(frameTime);
    keyboardEvent(frameTime);
    updates(frameTime);

    for(auto& object: object3D){
        object->animationTimer += timeScale * frameTime;
        object->updateAnimation(frameNumber);
    }
}

void scene::destroyScene()
{
    for(auto& graph: graphics){
        for(auto& lightSource: lightSources)    graph->removeLightSource(lightSource);
        for (auto& object: skyboxObjects)       graph->removeObject(object);
        for (auto& object: object3D)            graph->removeObject(object);
        for (auto& object: staticObject3D)      graph->removeObject(object);
        for (auto& model: models)               graph->destroyModel(model);
    }
    for(auto& lightPoint: lightPoints) delete lightPoint;
    lightPoints.clear();
    for(auto& object: skyboxObjects) delete object;
    skyboxObjects.clear();
    for(auto& object: object3D) delete object;
    object3D.clear();
    for(auto& object: staticObject3D) delete object;
    staticObject3D.clear();
    for(auto& model: models) delete model;
    models.clear();
}

void scene::loadModels()
{
    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/Bee.glb", 6));
    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/Box.glb"));
    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glTF/Sponza/Sponza.gltf"));
    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/Duck.glb"));
    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/RetroUFO.glb"));
    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glTF/Sponza/Sponza.gltf"));
    models.push_back(new class plyModel(ExternalPath / "dependences/model/pyramid.ply"));

    for(auto& model: models){
        graphics[0]->createModel(model);
    }
}

void scene::createLight()
{
    std::filesystem::path LIGHT_TEXTURE0  = ExternalPath / "dependences/texture/icon.PNG";
    std::filesystem::path LIGHT_TEXTURE1  = ExternalPath / "dependences/texture/light1.jpg";
    std::filesystem::path LIGHT_TEXTURE2  = ExternalPath / "dependences/texture/light2.jpg";
    std::filesystem::path LIGHT_TEXTURE3  = ExternalPath / "dependences/texture/light3.jpg";

    matrix<float,4,4> proj = perspective(radians(spotAngle), 1.0f, 0.1f, 20.0f);

    lightPoints.push_back(new isotropicLight(lightSources));
    lightPoints.back()->setLightColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(lightPoints.back());

    lightSources.push_back(new spotLight(LIGHT_TEXTURE0, proj, true, true));
    groups.at(2)->addObject(lightSources.back());

    lightSources.push_back(new spotLight(LIGHT_TEXTURE1, proj, true, true));
    groups.at(3)->addObject(lightSources.back());

    lightSources.push_back(new spotLight(LIGHT_TEXTURE2, proj, true, true));
    groups.at(4)->addObject(lightSources.back());

    lightSources.push_back(new spotLight(LIGHT_TEXTURE3, proj, true, true));
    groups.at(5)->addObject(lightSources.back());

    for(auto& graph: graphics){
        for(auto& source: lightSources){
            graph->bindLightSource(source, &graph == &graphics[0]);
        }
    }
}

void scene::createObjects()
{
    object3D.push_back( new baseObject(models.at(0), 0, 3));
    object3D.back()->setBloomColor(vector<float,4>(1.0,1.0,1.0,1.0));
    object3D.back()->translate(vector<float,3>(3.0f,0.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    object3D.back()->scale(vector<float,3>(0.2f,0.2f,0.2f));

    object3D.push_back(new baseObject(models.at(0), 3, 3));
    object3D.back()->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.7f));
    object3D.back()->translate(vector<float,3>(-3.0f,0.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    object3D.back()->scale(vector<float,3>(0.2f,0.2f,0.2f));
    object3D.back()->animationTimer = 1.0f;
    object3D.back()->animationIndex = 1;

    object3D.push_back(new baseObject(models.at(3)));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    object3D.back()->scale(vector<float,3>(3.0f));
    object3D.back()->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.8f));
    object3D.back()->animationTimer = 0.0f;
    object3D.back()->animationIndex = 0;
    groups.at(1)->addObject(object3D.back());

    staticObject3D.push_back(new baseObject(models.at(2)));
    staticObject3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    staticObject3D.back()->scale(vector<float,3>(3.0f,3.0f,3.0f));

    object3D.push_back(new baseObject(models.at(1)));
    object3D.back()->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,0.0f));
    object3D.back()->setBloomColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(0.0f,0.0f,1.0f,-0.8f));
    object3D.back()->setBloomFactor(vector<float,4>(1.0f,0.0f,0.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    groups.at(2)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(1.0f,0.0f,0.0f,-0.8f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    groups.at(3)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(1.0f,1.0f,0.0f,-0.8f));
    object3D.back()->setBloomFactor(vector<float,4>(0.0f,0.0f,1.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    groups.at(4)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(0.0f,1.0f,1.0f,-0.8f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    groups.at(5)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(6)));
    object3D.back()->translate(vector<float,3>(0.0f,0.0f,15.0f));
    object3D.back()->setConstantColor(vector<float,4>(0.2f,0.8f,1.0f,1.0f));
    object3D.back()->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,1.0f));

    for(auto& graph: graphics){
        for(auto& object: object3D){
            graph->bindObject(object, &graphics[0] == &graph);
        }
        for(auto& object: staticObject3D){
            graph->bindObject(object, &graphics[0] == &graph);
        }
    }
}

void scene::mouseEvent(float)
{
    int primitiveNumber = INT_FAST32_MAX;
    for(uint32_t i=0; i < app->getImageCount(); i++){
        primitiveNumber = graphics[0]->readStorageBuffer(i);
        if(primitiveNumber!=INT_FAST32_MAX)
            break;
    }

    glfwSetScrollCallback(window,[](GLFWwindow*, double, double yoffset) {
      spotAngle -= static_cast<float>(yoffset);
      updateLightCone = yoffset!=0.0;
    });

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        float sensitivity = 0.001;

        double x, y;
        glfwGetCursorPos(window,&x,&y);
        cameras->rotateX(sensitivity * static_cast<float>(yMpos - y), {1.0f,0.0f,0.0f});
        cameras->rotateY(sensitivity * static_cast<float>(xMpos - x), {0.0f,0.0f,1.0f});
        xMpos = x;
        yMpos = y;

        for(uint32_t i=0; i < app->getImageCount(); i++){
            graphics[0]->updateStorageBuffer(i, 2.0f * static_cast<float>(xMpos)/WIDTH - 1.0f , 2.0f * static_cast<float>(yMpos)/HEIGHT - 1.0f);
        }
    }
    else if(mouse1Stage == GLFW_PRESS && glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == 0)
    {
        for(auto& object: object3D){
            if(object->comparePrimitive(primitiveNumber)){
                controledObject = object;
                std::cout<< (&object - &object3D[0]) <<std::endl;
            }
        }
    }
    else
    {
        glfwGetCursorPos(window,&xMpos,&yMpos);
    }
    mouse1Stage = glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT);

    if(updateLightCone){
        for(auto& source: lightSources){
            if(spotAngle>0.0f){
                source->setProjectionMatrix(perspective(radians(spotAngle), 1.0f, 0.1f, 20.0f));
            }
        }
        updateLightCone = false;
    }
}

void scene::keyboardEvent(float frameTime)
{
    float sensitivity = 5.0f*frameTime;
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        cameras->translate(-sensitivity*cameras->getViewMatrix()[2].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        cameras->translate(sensitivity*cameras->getViewMatrix()[2].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        cameras->translate(-sensitivity*cameras->getViewMatrix()[0].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
    {
        cameras->translate(sensitivity*cameras->getViewMatrix()[0].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_Z) == GLFW_PRESS)
    {
        cameras->translate(sensitivity*cameras->getViewMatrix()[1].dvec());
    }
    if(glfwGetKey(window,GLFW_KEY_X) == GLFW_PRESS)
    {
        cameras->translate(-sensitivity*cameras->getViewMatrix()[1].dvec());
    }

    auto rotateControled = [this](const float& ang, const vector<float,3>& ax){
        if(bool foundInGroups = false; this->controledObject){
            for(auto& group: this->groups){
                if(group->findObject(this->controledObject)){
                    group->rotate(ang,ax);
                    foundInGroups = true;
                }
            }
            if(!foundInGroups) this->controledObject->rotate(ang,ax);
        }
    };

    auto translateControled = [this](const vector<float,3>& tr){
        if(bool foundInGroups = false; this->controledObject){
            for(auto& group: this->groups){
                if(group->findObject(this->controledObject)){
                    group->translate(tr);
                    foundInGroups = true;
                }
            }
            if(!foundInGroups) this->controledObject->translate(tr);
        }
    };

    if(glfwGetKey(window,GLFW_KEY_KP_4) == GLFW_PRESS)
    {
        rotateControled(radians(0.5f),{0.0f,0.0f,1.0f});
    }
    if(glfwGetKey(window,GLFW_KEY_KP_6) == GLFW_PRESS)
    {
        rotateControled(radians(-0.5f),{0.0f,0.0f,1.0f});
    }
    if(glfwGetKey(window,GLFW_KEY_KP_8) == GLFW_PRESS)
    {
        rotateControled(radians(0.5f),{1.0f,0.0f,0.0f});
    }
    if(glfwGetKey(window,GLFW_KEY_KP_5) == GLFW_PRESS)
    {
        rotateControled(radians(-0.5f),{1.0f,0.0f,0.0f});
    }
    if(glfwGetKey(window,GLFW_KEY_KP_7) == GLFW_PRESS)
    {
        rotateControled(radians(0.5f),{0.0f,1.0f,0.0f});
    }
    if(glfwGetKey(window,GLFW_KEY_KP_9) == GLFW_PRESS)
    {
        rotateControled(radians(-0.5f),{0.0f,1.0f,0.0f});
    }

    if(glfwGetKey(window,GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        translateControled(sensitivity*vector<float,3>(-1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        translateControled(sensitivity*vector<float,3>(1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS)
    {
        translateControled(sensitivity*vector<float,3>(0.0f,1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        translateControled(sensitivity*vector<float,3>(0.0f,-1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_ADD) == GLFW_PRESS)
    {
        translateControled(sensitivity*vector<float,3>(0.0f,0.0f,1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
    {
        translateControled(sensitivity*vector<float,3>(0.0f,0.0f,-1.0f));
    }

    if(glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window,GLFW_TRUE);
    }

    if(backOStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_O) == 0)
    {
        for(auto& object: object3D){
            if(controledObject == object){
                std::random_device device;
                std::uniform_real_distribution dist(0.3f, 1.0f);

                object->setOutlining(!object->getOutliningEnable(), 0.03f, {dist(device), dist(device), dist(device), dist(device)});
            }
        }
        graphics[0]->updateCmdFlags();
    }
    backOStage = glfwGetKey(window,GLFW_KEY_O);

    if(backTStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_T) == 0)
    {
        object3D[0]->changeAnimationFlag = true;
        object3D[0]->startTimer = object3D[0]->animationTimer;
        object3D[0]->changeAnimationTime = 0.5f;
        if(object3D[0]->animationIndex == 0)
            object3D[0]->newAnimationIndex = 1;
        else if(object3D[0]->animationIndex == 1)
            object3D[0]->newAnimationIndex = 0;
    }
    backTStage = glfwGetKey(window,GLFW_KEY_T);

    if(backNStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_N) == 0)
    {
        object3D.push_back(new baseObject(models.at(4)));
        object3D.back()->translate(cameras->getTranslation());
        object3D.back()->rotate(radians(90.0f),{1.0f,0.0f,0.0f});
        graphics[0]->bindObject(object3D.back(), true);
    }
    backNStage = glfwGetKey(window,GLFW_KEY_N);

    if(backBStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_B) == 0)
    {
        app->deviceWaitIdle();
        if(object3D.size() && graphics[0]->removeObject(object3D.back())) {
            object3D.pop_back();
        }
    }
    backBStage = glfwGetKey(window,GLFW_KEY_B);

    if(glfwGetKey(window,GLFW_KEY_KP_0) == GLFW_PRESS)
    {
        minAmbientFactor -= minAmbientFactor > 0.011 ? 0.01f : 0.0f;
        graphics[0]->setMinAmbientFactor(minAmbientFactor);
    }
    if(glfwGetKey(window,GLFW_KEY_KP_2) == GLFW_PRESS)
    {
        minAmbientFactor += 0.01f;
        graphics[0]->setMinAmbientFactor(minAmbientFactor);
    }

    if(glfwGetKey(window,GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
    {
        timeScale -= timeScale > 0.0051f ? 0.005f : 0.0f;
    }
    if(glfwGetKey(window,GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
    {
        timeScale += 0.005f;
    }
}

void scene::updates(float frameTime)
{
    globalTime += frameTime;

    skyboxObjects[1]->rotate(0.1f*frameTime,normalize(vector<float,3>(1.0f,1.0f,1.0f)));
}

