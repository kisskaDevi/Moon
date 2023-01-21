#include "scene.h"
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>
#include "core/graphics/deferredGraphics/deferredgraphicsinterface.h"

bool updateLightCone = false;
float spotAngle = 90.0f;

bool updateCamera = false;
float cameraAngle = 45.0f;

scene::scene(graphicsManager *app, deferredGraphicsInterface* graphics, std::string ExternalPath)
{
    this->app = app;
    this->graphics = graphics;
    this->ExternalPath = ExternalPath;

    ZERO_TEXTURE        = ExternalPath + "texture\\0.png";
    ZERO_TEXTURE_WHITE  = ExternalPath + "texture\\1.png";
}

void scene::createScene(uint32_t WIDTH, uint32_t HEIGHT, camera* cameraObject)
{
    this->WIDTH = WIDTH;
    this->HEIGHT = HEIGHT;

    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);

    std::vector<std::string> SKYBOX = {
        ExternalPath+"texture\\skybox\\left.jpg",
        ExternalPath+"texture\\skybox\\right.jpg",
        ExternalPath+"texture\\skybox\\front.jpg",
        ExternalPath+"texture\\skybox\\back.jpg",
        ExternalPath+"texture\\skybox\\top.jpg",
        ExternalPath+"texture\\skybox\\bottom.jpg"
    };

    std::vector<std::string> SKYBOX1 = {
        ExternalPath+"texture\\skybox1\\left.png",
        ExternalPath+"texture\\skybox1\\right.png",
        ExternalPath+"texture\\skybox1\\front.png",
        ExternalPath+"texture\\skybox1\\back.png",
        ExternalPath+"texture\\skybox1\\top.png",
        ExternalPath+"texture\\skybox1\\bottom.png"
    };

    cameras = cameraObject;

    skyboxObject1 = new skyboxObject(SKYBOX);
        skyboxObject1->scale(glm::vec3(200.0f,200.0f,200.0f));
    graphics->bindSkyBoxObject(skyboxObject1);
    skyboxObject1->setColorFactor(glm::vec4(0.5));

    skyboxObject2 = new skyboxObject(SKYBOX1);
        skyboxObject2->scale(glm::vec3(200.0f,200.0f,200.0f));
    graphics->bindSkyBoxObject(skyboxObject2);

    loadModels();
    createLight();
    createObjects();
}

void scene::updateFrame(GLFWwindow* window, uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT)
{
    this->WIDTH = WIDTH;
    this->HEIGHT = HEIGHT;

    glm::mat4x4 proj = glm::perspective(glm::radians(cameraAngle), (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
    proj[1][1] *= -1.0f;
    cameras->setProjMatrix(proj);

    glfwPollEvents();
    mouseEvent(window,frameTime);
    keyboardEvent(window,frameTime);
    updates(frameTime);

    for(size_t j=0;j<object3D.size();j++){
        object3D[j]->animationTimer += timeScale*frameTime;
        object3D[j]->updateAnimation(frameNumber);
    }
}

void scene::destroyScene()
{
    for(size_t i=0;i<lightSource.size();i++){
        graphics->removeLightSource(lightSource.at(i));
    }
    for(size_t i=0;i<lightPoint.size();i++)
        delete lightPoint.at(i);

    graphics->removeSkyBoxObject(skyboxObject1);
    delete skyboxObject1;

    graphics->removeSkyBoxObject(skyboxObject2);
    delete skyboxObject2;

    for (size_t i =0 ;i<gltfModel.size();i++)
        for (size_t j =0 ;j<gltfModel.at(i).size();j++)
            graphics->destroyModel(gltfModel.at(i)[j]);

    for (size_t i=0 ;i<object3D.size();i++){
        graphics->removeObject(object3D[i]);
        delete object3D.at(i);
    }

    graphics->destroyEmptyTextures();
}

void scene::loadModels()
{
    size_t index = 0;

    gltfModel.resize(6);

    index = 0;
        gltfModel[0].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        graphics->createModel(gltfModel[0].at(index));
    index++;
        gltfModel[0].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        graphics->createModel(gltfModel[0].at(index));
    index++;
        gltfModel[0].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        graphics->createModel(gltfModel[0].at(index));
    index++;

    index = 0;
        gltfModel[1].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        graphics->createModel(gltfModel[1].at(index));
    index++;
        gltfModel[1].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        graphics->createModel(gltfModel[1].at(index));
    index++;
        gltfModel[1].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        graphics->createModel(gltfModel[1].at(index));
    index++;

    index = 0;
        gltfModel[2].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Box.glb"));
        graphics->createModel(gltfModel[2].at(index));
    index++;

    index = 0;
        gltfModel[3].push_back(new struct gltfModel(ExternalPath + "model\\glTF\\Sponza\\Sponza.gltf"));
        graphics->createModel(gltfModel[3].at(index));
    index++;

    index = 0;
        gltfModel[4].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Duck.glb"));
        graphics->createModel(gltfModel[4].at(index));
    index++;

    index = 0;
        gltfModel[5].push_back(new struct gltfModel(ExternalPath + "model\\glb\\RetroUFO.glb"));
        graphics->createModel(gltfModel[5].at(index));
    index++;
}

void scene::createLight()
{
    std::string LIGHT_TEXTURE0  = ExternalPath + "texture\\icon.PNG";
    std::string LIGHT_TEXTURE1  = ExternalPath + "texture\\light1.jpg";
    std::string LIGHT_TEXTURE2  = ExternalPath + "texture\\light2.jpg";
    std::string LIGHT_TEXTURE3  = ExternalPath + "texture\\light3.jpg";

    glm::mat4x4 Proj = glm::perspective(glm::radians(spotAngle), 1.0f, 0.1f, 100.0f);
    Proj[1][1] *= -1;

    int index = 0;
    lightPoint.push_back(new isotropicLight(lightSource));
    lightPoint.at(index)->setProjectionMatrix(Proj);
    lightPoint.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(lightPoint.at(index));

    for(int i=index;i<6;i++,index++){
        graphics->bindLightSource(lightSource.at(index));
        //lightSource.at(index)->setScattering(true);
    }

    Proj = glm::perspective(glm::radians(spotAngle), 1.0f, 0.1f, 20.0f);
    Proj[1][1] *= -1;

    lightSource.push_back(new spotLight(LIGHT_TEXTURE0));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(2)->addObject(lightSource.at(index));
    index++;
    graphics->bindLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new spotLight(LIGHT_TEXTURE1));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(3)->addObject(lightSource.at(index));
    index++;
    graphics->bindLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new spotLight(LIGHT_TEXTURE2));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(4)->addObject(lightSource.at(index));
    index++;
    graphics->bindLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new spotLight(LIGHT_TEXTURE3));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(5)->addObject(lightSource.at(index));
    index++;
    graphics->bindLightSource(lightSource.at(lightSource.size()-1));

    for(int i=0;i<5;i++){
        lightSource.push_back(new spotLight(LIGHT_TEXTURE0));
        lightSource.at(index)->setProjectionMatrix(Proj);
        lightSource.at(index)->translate(glm::vec3(20.0f-10.0f*i,10.0f,3.0f));
        lightSource.at(index)->setScattering(false);
        index++;
    }

    for(int i=0;i<5;i++){
        lightSource.push_back(new spotLight(LIGHT_TEXTURE0));
        lightSource.at(index)->setProjectionMatrix(Proj);
        lightSource.at(index)->translate(glm::vec3(20.0f-10.0f*i,-10.0f,3.0f));
        lightSource.at(index)->setScattering(false);
        index++;
    }
}

void scene::createObjects()
{
    uint32_t index=0;
    object3D.push_back( new object(gltfModel.at(0).size(),gltfModel.at(0).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setOutliningColor(glm::vec4(0.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->setOutliningWidth(0.05f);
    object3D.at(index)->setBloomColor(glm::vec4(1.0,1.0,1.0,1.0));
    object3D.at(index)->translate(glm::vec3(3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    index++;

    object3D.push_back( new object(gltfModel.at(1).size(),gltfModel.at(1).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setOutliningColor(glm::vec4(1.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->setOutliningWidth(0.05f);
    object3D.at(index)->setConstantColor(glm::vec4(1.0,0.0,0.0,-0.8));
    object3D.at(index)->translate(glm::vec3(-3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    object3D.at(index)->animationTimer = 1.0f;
    object3D.at(index)->animationIndex = 1;
    index++;

    object3D.push_back( new object(gltfModel.at(4).size(),gltfModel.at(4).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setOutliningColor(glm::vec4(0.7f,0.5f,0.2f,1.0f));
    object3D.at(index)->setOutliningWidth(0.025f);
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(3.0f));
    object3D.at(index)->setConstantColor(glm::vec4(0.0f,0.0f,0.0f,-0.8f));
    object3D.at(index)->animationTimer = 0.0f;
    object3D.at(index)->animationIndex = 0;
    object *Duck = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(3).size(),gltfModel.at(3).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(3.0f,3.0f,3.0f));
    index++;

    object3D.push_back( new object(gltfModel.at(2).size(),gltfModel.at(2).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColorFactor(glm::vec4(0.0f,0.0f,0.0f,0.0f));
    object3D.at(index)->setBloomColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    object *Box0 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setConstantColor(glm::vec4(0.0f,0.0f,1.0f,-0.8f));
    object3D.at(index)->setBloomFactor(glm::vec4(1.0f,0.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO1 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setConstantColor(glm::vec4(1.0f,0.0f,0.0f,-0.8f));
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO2 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setConstantColor(glm::vec4(1.0f,1.0f,0.0f,-0.8f));
    object3D.at(index)->setBloomFactor(glm::vec4(0.0f,0.0f,1.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO3 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setConstantColor(glm::vec4(0.0f,1.0f,1.0f,-0.8f));
    object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO4 = object3D.at(index);
    index++;

    groups.at(0)->translate(glm::vec3(0.0f,0.0f,5.0f));
    groups.at(0)->addObject(Box0);

    groups.at(1)->translate(glm::vec3(0.0f,0.0f,3.0f));
    groups.at(1)->addObject(Duck);

    groups.at(2)->translate(glm::vec3(5.0f,0.0f,5.0f));
    groups.at(2)->addObject(UFO1);

    groups.at(3)->translate(glm::vec3(-5.0f,0.0f,5.0f));
    groups.at(3)->addObject(UFO2);

    groups.at(4)->translate(glm::vec3(10.0f,0.0f,5.0f));
    groups.at(4)->addObject(UFO3);

    groups.at(5)->translate(glm::vec3(-10.0f,0.0f,5.0f));
    groups.at(5)->addObject(UFO4);
}

void scene::mouseEvent(GLFWwindow* window, float frameTime)
{
    static_cast<void>(frameTime);

    double x, y;

    int primitiveNumber = INT_FAST32_MAX;
    for(uint32_t i=0;i<graphics->getImageCount();i++){
        primitiveNumber = graphics->readStorageBuffer(i);
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
        cameras->rotateX(angy,glm::vec3(1.0f,0.0f,0.0f));
        cameras->rotateY(angx,glm::vec3(0.0f,0.0f,1.0f));

        for(uint32_t i=0;i<graphics->getImageCount();i++){
            graphics->updateStorageBuffer(i, -1.0f+2.0f*xMpos/(WIDTH), -1.0f+2.0f*yMpos/(HEIGHT));
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
        for(uint32_t index=6;index<lightSource.size();index++){
            if(spotAngle>0.0f){
                glm::mat4x4 Proj;
                    Proj = glm::perspective(glm::radians(spotAngle), 1.0f, 0.1f, 20.0f);
                    Proj[1][1] *= -1;
                lightSource.at(index)->setProjectionMatrix(Proj);
            }
        }
        updateLightCone = false;
    }

    if(updateCamera){
        if(cameraAngle>0.0f){
            glm::mat4x4 proj = glm::perspective(glm::radians(cameraAngle), (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
            proj[1][1] *= -1.0f;
            cameras->setProjMatrix(proj);
        }
        updateCamera = false;
    }
}

void scene::keyboardEvent(GLFWwindow* window, float frameTime)
{
    float sensitivity = 5.0f*frameTime;
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[0][2];
        float y = -sensitivity*cameras->getViewMatrix()[1][2];
        float z = -sensitivity*cameras->getViewMatrix()[2][2];
        cameras->translate(glm::vec3(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[0][2];
        float y = sensitivity*cameras->getViewMatrix()[1][2];
        float z = sensitivity*cameras->getViewMatrix()[2][2];
        cameras->translate(glm::vec3(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[0][0];
        float y = -sensitivity*cameras->getViewMatrix()[1][0];
        float z = -sensitivity*cameras->getViewMatrix()[2][0];
        cameras->translate(glm::vec3(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[0][0];
        float y = sensitivity*cameras->getViewMatrix()[1][0];
        float z = sensitivity*cameras->getViewMatrix()[2][0];
        cameras->translate(glm::vec3(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_Z) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[0][1];
        float y = sensitivity*cameras->getViewMatrix()[1][1];
        float z = sensitivity*cameras->getViewMatrix()[2][1];
        cameras->translate(glm::vec3(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_X) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[0][1];
        float y = -sensitivity*cameras->getViewMatrix()[1][1];
        float z = -sensitivity*cameras->getViewMatrix()[2][1];
        cameras->translate(glm::vec3(x,y,z));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_4) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(0.0f,0.0f,1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_6) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(0.0f,0.0f,1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_8) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_5) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_7) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(0.0f,1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_9) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(0.0f,1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(-1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(1.0f,0.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,-1.0f,0.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_ADD) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,0.0f,1.0f));
    }
    if(glfwGetKey(window,GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,0.0f,-1.0f));
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
    if(backRStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_R) == 0)
    {
        framebufferResized = true;
    }
    backRStage = glfwGetKey(window,GLFW_KEY_R);
    if(backOStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_O) == 0)
    {
        object3D[0]->setOutliningEnable(!object3D[0]->getOutliningEnable());
        object3D[1]->setOutliningEnable(!object3D[1]->getOutliningEnable());
        object3D[2]->setOutliningEnable(!object3D[2]->getOutliningEnable());
        graphics->updateCmdFlags();
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
    if(backYStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_Y) == 0)
    {
        object3D.at(2)->changeAnimationFlag = true;
        object3D.at(2)->startTimer = object3D.at(2)->animationTimer;
        object3D.at(2)->changeAnimationTime = 0.1f;
        if(object3D.at(2)->animationIndex<4)
            object3D.at(2)->newAnimationIndex += 1;
        else
            object3D.at(2)->newAnimationIndex = 0;
    }
    backYStage = glfwGetKey(window,GLFW_KEY_Y);
    if(backNStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_N) == 0)
    {
        size_t index = object3D.size();
        object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
        graphics->bindBaseObject(object3D.at(index));
        object3D.at(index)->translate(cameras->getTranslation());
        object3D.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    }
    backNStage = glfwGetKey(window,GLFW_KEY_N);
    if(backBStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_B) == 0)
    {
        app->deviceWaitIdle();
        if(object3D.size()>0){
            size_t index = object3D.size()-1;
            if(graphics->removeObject(object3D[index]))
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
            graphics->bindLightSource(lightSource.at(lightPointer));
            lightPointer++;
        }
    }
    backGStage = glfwGetKey(window,GLFW_KEY_G);
    if(backHStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_H) == 0)
    {
        app->deviceWaitIdle();
        if(lightPointer>0){
            lightPointer--;
            graphics->removeLightSource(lightSource.at(lightPointer));
        }
    }
    backHStage = glfwGetKey(window,GLFW_KEY_H);

    if(glfwGetKey(window,GLFW_KEY_KP_0) == GLFW_PRESS)
    {
        minAmbientFactor -= 0.1f*sensitivity;
        graphics->setMinAmbientFactor(minAmbientFactor);
    }
    if(glfwGetKey(window,GLFW_KEY_KP_2) == GLFW_PRESS)
    {
        minAmbientFactor += 0.1f*sensitivity;
        graphics->setMinAmbientFactor(minAmbientFactor);
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

    skyboxObject2->rotate(0.1f*frameTime,glm::normalize(glm::vec3(1.0f,1.0f,1.0f)));
}

void scrol(GLFWwindow *window, double xoffset, double yoffset)
{
    static_cast<void>(window);

    spotAngle -= yoffset;
    if(yoffset!=0.0) updateLightCone = true;

    cameraAngle -= xoffset;
    if(xoffset!=0.0) updateCamera = true;
}
