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

bool updateLightCone = false;
float spotAngle = 90.0f;

bool updateCamera = false;
float cameraAngle = 45.0f;

scene::scene(graphicsManager *app, std::vector<deferredGraphics*> graphics, GLFWwindow* window, const std::filesystem::path& ExternalPath):
    app(app),
    graphics(graphics),
    window(window),
    ExternalPath(ExternalPath),
    ZERO_TEXTURE(ExternalPath / "dependences/texture/0.png"),
    ZERO_TEXTURE_WHITE(ExternalPath / "dependences/texture/1.png")
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

    skyboxObject1 = new skyboxObject(SKYBOX);
    skyboxObject1->scale(vector<float,3>(200.0f,200.0f,200.0f));
    skyboxObject1->setColorFactor(vector<float,4>(0.5));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(skyboxObject1, i == 0);
    }

    skyboxObject2 = new skyboxObject(SKYBOX1);
    skyboxObject2->scale(vector<float,3>(200.0f,200.0f,200.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(skyboxObject2, i == 0);
    }

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
    for(auto& lightSource: lightSources){
        for(size_t i = 0; i < graphics.size(); i++){
            graphics[i]->removeLightSource(lightSource);
        }
    }

    for(auto& lightPoint: lightPoints){
        delete lightPoint;
    }

    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->removeObject(skyboxObject1);
    }
    delete skyboxObject1;

    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->removeObject(skyboxObject2);
    }
    delete skyboxObject2;

    for (auto& model: models){
        graphics[0]->destroyModel(model);
    }

    for (auto& object: object3D){
        for(size_t i = 0; i < graphics.size(); i++){
            graphics[i]->removeObject(object);
        }
        delete object;
    }
}

void scene::loadModels()
{
    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/Bee.glb", 6));
    graphics[0]->createModel(models.back());

    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/Box.glb"));
    graphics[0]->createModel(models.back());

    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glTF/Sponza/Sponza.gltf"));
    graphics[0]->createModel(models.back());

    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/Duck.glb"));
    graphics[0]->createModel(models.back());

    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glb/RetroUFO.glb"));
    graphics[0]->createModel(models.back());

    models.push_back(new class gltfModel(ExternalPath / "dependences/model/glTF/Sponza/Sponza.gltf"));
    graphics[0]->createModel(models.back());

    models.push_back(new class plyModel(ExternalPath / "dependences/model/plytest.ply"));
    graphics[0]->createModel(models.back());

    models.push_back(new class plyModel(ExternalPath / "dependences/model/pyramid.ply"));
    graphics[0]->createModel(models.back());
}

void scene::createLight()
{
    std::filesystem::path LIGHT_TEXTURE0  = ExternalPath / "dependences/texture/icon.PNG";
    std::filesystem::path LIGHT_TEXTURE1  = ExternalPath / "dependences/texture/light1.jpg";
    std::filesystem::path LIGHT_TEXTURE2  = ExternalPath / "dependences/texture/light2.jpg";
    std::filesystem::path LIGHT_TEXTURE3  = ExternalPath / "dependences/texture/light3.jpg";

    matrix<float,4,4> Proj = perspective(radians(91.0f), 1.0f, 0.1f, 100.0f);

    int index = 0;
    lightPoints.push_back(new isotropicLight(lightSources));
    lightPoints.at(index)->setProjectionMatrix(Proj);
    lightPoints.at(index)->setLightColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(lightPoints.at(index));

    for(int i=index;i<6;i++,index++){
        for(size_t i = 0; i < graphics.size(); i++){
            graphics[i]->bindLightSource(lightSources.at(index), i == 0);
        }
    }

    Proj = perspective(radians(spotAngle), 1.0f, 0.1f, 20.0f);

    lightSources.push_back(new spotLight(LIGHT_TEXTURE0));
    lightSources.at(index)->setProjectionMatrix(Proj);
    lightSources.at(index)->setScattering(true);
    groups.at(2)->addObject(lightSources.at(index));
    index++;
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindLightSource(lightSources.at(lightSources.size()-1), i == 0);
    }

    lightSources.push_back(new spotLight(LIGHT_TEXTURE1));
    lightSources.at(index)->setProjectionMatrix(Proj);
    lightSources.at(index)->setScattering(true);
    groups.at(3)->addObject(lightSources.at(index));
    index++;
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindLightSource(lightSources.at(lightSources.size()-1), i == 0);
    }

    lightSources.push_back(new spotLight(LIGHT_TEXTURE2));
    lightSources.at(index)->setProjectionMatrix(Proj);
    lightSources.at(index)->setScattering(true);
    groups.at(4)->addObject(lightSources.at(index));
    index++;
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindLightSource(lightSources.at(lightSources.size()-1), i == 0);
    }

    lightSources.push_back(new spotLight(LIGHT_TEXTURE3));
    lightSources.at(index)->setProjectionMatrix(Proj);
    lightSources.at(index)->setScattering(true);
    groups.at(5)->addObject(lightSources.at(index));
    index++;
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindLightSource(lightSources.at(lightSources.size()-1), i == 0);
    }

    for(int i=0;i<5;i++){
        lightSources.push_back(new spotLight(LIGHT_TEXTURE0));
        lightSources.at(index)->setProjectionMatrix(Proj);
        lightSources.at(index)->translate(vector<float,3>(20.0f-10.0f*i,10.0f,3.0f));
        lightSources.at(index)->setScattering(false);
        index++;
    }

    for(int i=0;i<5;i++){
        lightSources.push_back(new spotLight(LIGHT_TEXTURE0));
        lightSources.at(index)->setProjectionMatrix(Proj);
        lightSources.at(index)->translate(vector<float,3>(20.0f-10.0f*i,-10.0f,3.0f));
        lightSources.at(index)->setScattering(false);
        index++;
    }
}

void scene::createObjects()
{
    object3D.push_back( new baseObject(models.at(0), 0, 3));
    object3D.back()->setOutliningColor(vector<float,4>(0.0f,0.5f,0.8f,1.0f));
    object3D.back()->setOutliningWidth(0.05f);
    object3D.back()->setBloomColor(vector<float,4>(1.0,1.0,1.0,1.0));
    object3D.back()->translate(vector<float,3>(3.0f,0.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    object3D.back()->scale(vector<float,3>(0.2f,0.2f,0.2f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }

    object3D.push_back(new baseObject(models.at(0), 3, 3));
    object3D.back()->setOutliningColor(vector<float,4>(1.0f,0.5f,0.8f,1.0f));
    object3D.back()->setOutliningWidth(0.05f);
    object3D.back()->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.7f));
    object3D.back()->translate(vector<float,3>(-3.0f,0.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    object3D.back()->scale(vector<float,3>(0.2f,0.2f,0.2f));
    object3D.back()->animationTimer = 1.0f;
    object3D.back()->animationIndex = 1;
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }

    object3D.push_back(new baseObject(models.at(3)));
    object3D.back()->setOutliningColor(vector<float,4>(0.7f,0.5f,0.2f,1.0f));
    object3D.back()->setOutliningWidth(0.025f);
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    object3D.back()->scale(vector<float,3>(3.0f));
    object3D.back()->setConstantColor(vector<float,4>(0.0f,0.0f,0.0f,-0.8f));
    object3D.back()->animationTimer = 0.0f;
    object3D.back()->animationIndex = 0;
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }
    groups.at(1)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(2)));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    object3D.back()->scale(vector<float,3>(3.0f,3.0f,3.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }

    object3D.push_back(new baseObject(models.at(1)));
    object3D.back()->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,0.0f));
    object3D.back()->setBloomColor(vector<float,4>(1.0f,1.0f,1.0f,1.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }
    groups.at(0)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(0.0f,0.0f,1.0f,-0.8f));
    object3D.back()->setBloomFactor(vector<float,4>(1.0f,0.0f,0.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }
    groups.at(2)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(1.0f,0.0f,0.0f,-0.8f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }
    groups.at(3)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(1.0f,1.0f,0.0f,-0.8f));
    object3D.back()->setBloomFactor(vector<float,4>(0.0f,0.0f,1.0f,0.0f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }
    groups.at(4)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(4)));
    object3D.back()->setConstantColor(vector<float,4>(0.0f,1.0f,1.0f,-0.8f));
    object3D.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }
    groups.at(5)->addObject(object3D.back());

    object3D.push_back(new baseObject(models.at(6)));
    object3D.back()->scale(vector<float,3>(0.002f));
    object3D.back()->translate(vector<float,3>(0.0f,0.0f,15.0f));
    object3D.back()->setConstantColor(vector<float,4>(0.2f,0.8f,1.0f,1.0f));
    object3D.back()->setColorFactor(vector<float,4>(0.0f,0.0f,0.0f,1.0f));
    for(size_t i = 0; i < graphics.size(); i++){
        graphics[i]->bindObject(object3D.back(), i == 0);
    }
}

void scene::mouseEvent(float frameTime)
{
    static_cast<void>(frameTime);

    double x, y;

    int primitiveNumber = INT_FAST32_MAX;
    for(uint32_t i=0; i < app->getImageCount(); i++){
        primitiveNumber = graphics[0]->readStorageBuffer(i);
        if(primitiveNumber!=INT_FAST32_MAX)
            break;
    }

    glfwSetScrollCallback(window,scrol);

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        double sensitivity = 0.001;
        glfwGetCursorPos(window,&x,&y);
        angx = sensitivity*(x - xMpos);
        angy = sensitivity*(y - yMpos);
        xMpos = x;
        yMpos = y;
        cameras->rotateX(static_cast<float>(-angy), vector<float,3>(1.0f,0.0f,0.0f));
        cameras->rotateY(static_cast<float>(angx), vector<float,3>(0.0f,0.0f,1.0f));

        for(uint32_t i=0; i < app->getImageCount(); i++){
            graphics[0]->updateStorageBuffer(i, 2.0f * static_cast<float>(xMpos)/WIDTH - 1.0f , 2.0f * static_cast<float>(yMpos)/HEIGHT - 1.0f);
        }
    }
    else if(mouse1Stage == GLFW_PRESS && glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == 0)
    {
        uint32_t index = 0;
        for(auto object: object3D){
            if(object->comparePrimitive(primitiveNumber)){
                std::cout<<index<<std::endl;
            }
            index++;
        }
    }
    else
    {
        glfwGetCursorPos(window,&xMpos,&yMpos);
    }
    mouse1Stage = glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT);

    if(updateLightCone){
        for(uint32_t index=6;index<lightSources.size();index++){
            if(spotAngle>0.0f){
                matrix<float,4,4> Proj;
                    Proj = perspective(radians(spotAngle), 1.0f, 0.1f, 20.0f);
                    Proj[1][1] *= -1;
                lightSources.at(index)->setProjectionMatrix(Proj);
            }
        }
        updateLightCone = false;
    }

    if(updateCamera){
        if(cameraAngle>0.0f){
            cameras->recreate(cameraAngle, (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
        }
        updateCamera = false;
    }
}

void scene::keyboardEvent(float frameTime)
{
    float sensitivity = 5.0f*frameTime;
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[2][0];
        float y = -sensitivity*cameras->getViewMatrix()[2][1];
        float z = -sensitivity*cameras->getViewMatrix()[2][2];
        cameras->translate(vector<float,3>(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[2][0];
        float y = sensitivity*cameras->getViewMatrix()[2][1];
        float z = sensitivity*cameras->getViewMatrix()[2][2];
        cameras->translate(vector<float,3>(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[0][0];
        float y = -sensitivity*cameras->getViewMatrix()[0][1];
        float z = -sensitivity*cameras->getViewMatrix()[0][2];
        cameras->translate(vector<float,3>(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[0][0];
        float y = sensitivity*cameras->getViewMatrix()[0][1];
        float z = sensitivity*cameras->getViewMatrix()[0][2];
        cameras->translate(vector<float,3>(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_Z) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[1][0];
        float y = sensitivity*cameras->getViewMatrix()[1][1];
        float z = sensitivity*cameras->getViewMatrix()[1][2];
        cameras->translate(vector<float,3>(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_X) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[1][0];
        float y = -sensitivity*cameras->getViewMatrix()[1][1];
        float z = -sensitivity*cameras->getViewMatrix()[1][2];
        cameras->translate(vector<float,3>(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_4) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(radians(0.5f),vector<float,3>(0.0f,0.0f,1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_6) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(radians(-0.5f),vector<float,3>(0.0f,0.0f,1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_8) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(radians(0.5f),vector<float,3>(1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_5) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(radians(-0.5f),vector<float,3>(1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_7) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(radians(0.5f),vector<float,3>(0.0f,1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_9) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(radians(-0.5f),vector<float,3>(0.0f,1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*vector<float,3>(-1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*vector<float,3>(1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*vector<float,3>(0.0f,1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*vector<float,3>(0.0f,-1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_ADD) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*vector<float,3>(0.0f,0.0f,1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*vector<float,3>(0.0f,0.0f,-1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_1) == GLFW_PRESS)
    {
        controledGroup = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_2) == GLFW_PRESS)
    {
        controledGroup = 1;
    }
    if(glfwGetKey(window,GLFW_KEY_3) == GLFW_PRESS)
    {
        controledGroup = 2;
    }
    if(glfwGetKey(window,GLFW_KEY_4) == GLFW_PRESS)
    {
        controledGroup = 3;
    }
    if(glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window,GLFW_TRUE);
    }
    if(backOStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_O) == 0)
    {
        object3D[0]->setOutliningEnable(!object3D[0]->getOutliningEnable());
        object3D[1]->setOutliningEnable(!object3D[1]->getOutliningEnable());
        object3D[2]->setOutliningEnable(!object3D[2]->getOutliningEnable());
        //object3D[10]->setOutliningEnable(!object3D[10]->getOutliningEnable());
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
        size_t index = object3D.size();
        object3D.push_back(new baseObject(models.at(4)));
        graphics[0]->bindObject(object3D.at(index));
        object3D.at(index)->translate(cameras->getTranslation());
        object3D.at(index)->rotate(radians(-90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    }
    backNStage = glfwGetKey(window,GLFW_KEY_N);
    if(backBStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_B) == 0)
    {
        app->deviceWaitIdle();
        if(object3D.size()>0){
            size_t index = object3D.size()-1;
            if(graphics[0]->removeObject(object3D[index]))
            {
                delete object3D[index];
                object3D.erase(object3D.begin()+index);
            }
        }
    }
    backBStage = glfwGetKey(window,GLFW_KEY_B);
    if(backGStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_G) == 0)
    {
        if(lightPointer<20){
            graphics[0]->bindLightSource(lightSources.at(lightPointer));
            lightPointer++;
        }
    }
    backGStage = glfwGetKey(window,GLFW_KEY_G);
    if(backHStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_H) == 0)
    {
        app->deviceWaitIdle();
        if(lightPointer>0){
            lightPointer--;
            graphics[0]->removeLightSource(lightSources.at(lightPointer));
        }
    }
    backHStage = glfwGetKey(window,GLFW_KEY_H);

    if(glfwGetKey(window,GLFW_KEY_KP_0) == GLFW_PRESS)
    {
        minAmbientFactor -= 0.1f*sensitivity;
        graphics[0]->setMinAmbientFactor(minAmbientFactor);
    }
    if(glfwGetKey(window,GLFW_KEY_KP_2) == GLFW_PRESS)
    {
        minAmbientFactor += 0.1f*sensitivity;
        graphics[0]->setMinAmbientFactor(minAmbientFactor);
    }

    if(glfwGetKey(window,GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
    {
        if(timeScale>0.0051f){
            timeScale -= 0.005f;
        }
    }

    if(glfwGetKey(window,GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
    {
        timeScale += 0.005f;
    }
}

void scene::updates(float frameTime)
{
    globalTime += frameTime;

    skyboxObject2->rotate(0.1f*frameTime,normalize(vector<float,3>(1.0f,1.0f,1.0f)));
}

void scrol(GLFWwindow *window, double xoffset, double yoffset)
{
    static_cast<void>(window);

    spotAngle -= static_cast<float>(yoffset);
    if(yoffset!=0.0) updateLightCone = true;

    cameraAngle -= static_cast<float>(xoffset);
    if(xoffset!=0.0) updateCamera = true;
}
