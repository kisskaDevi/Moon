#ifdef TestScene1
std::string LIGHT_TEXTURE  = ExternalPath + "texture\\icon.PNG";

float frameTime;
float fps = 60.0f;
bool  animate = true;
bool  fpsLock = false;

double   xMpos, yMpos;
double   angx=0.0, angy=0.0;
int mouse1Stage = 0;
int spaceStage = 0;

float spotAngle = 90.0f;
bool updateLightCone = false;

float minAmbientFactor = 0.05f;

uint32_t controledGroup = 0;

bool     backRStage = 0;
bool     backTStage = 0;
bool     backYStage = 0;
bool     backNStage = 0;
bool     backBStage = 0;
bool     backOStage = 0;
bool     backIStage = 0;
bool     backGStage = 0;
bool     backHStage = 0;

uint32_t lightPointer = 10;

void scrol(GLFWwindow* window, double xoffset, double yoffset);
void mouseEvent(VkApplication* app, GLFWwindow* window, float frameTime);
void keyboardEvent(VkApplication* app, GLFWwindow* window, float frameTime);

void loadModels(VkApplication* app);
void createLight(VkApplication* app);
void createObjects(VkApplication* app);

std::vector<std::vector<gltfModel*>>         gltfModel;
std::vector<object              *>          object3D;
std::vector<light<spotLight>    *>          lightSource;
std::vector<light<pointLight>   *>          lightPoint;
std::vector<group               *>          groups;

object *skyboxObject;

void createScene(VkApplication *app)
{
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);
    groups.push_back(new group);

    skyboxObject = new object(1,nullptr);
        skyboxObject->scale(glm::vec3(200.0f,200.0f,200.0f));
    app->bindSkyBoxObject(skyboxObject, SKYBOX);


    loadModels(app);
    createLight(app);
    createObjects(app);
}

void runScene(VkApplication *app, GLFWwindow* window)
{
    static auto pastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        frameTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - pastTime).count();

            if(fpsLock)
                if(fps<1.0f/frameTime)  continue;
            pastTime = currentTime;

            std::stringstream ss;
            ss << "Vulkan" << " [" << 1.0f/frameTime << " FPS]";
            glfwSetWindowTitle(window, ss.str().c_str());


        glfwPollEvents();
        mouseEvent(app,window,frameTime);
        keyboardEvent(app,window,frameTime);

        VkResult result = app->drawFrame(frameTime, object3D);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)                         recreateSwapChain(app,window);
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)   throw std::runtime_error("failed to acquire swap chain image!");

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
            framebufferResized = false;
            recreateSwapChain(app,window);
            glm::mat4x4 proj = glm::perspective(glm::radians(45.0f), (float) WIDTH / (float) HEIGHT, 0.1f, 1000.0f);
            proj[1][1] *= -1.0f;
            cameras->setProjMatrix(proj);
        }else if(result != VK_SUCCESS){
            throw std::runtime_error("failed to present swap chain image!");
        }

        //app->deviceWaitIdle();  //???костыль
    }
}

void destroyScene(VkApplication *app)
{
    for(size_t i=0;i<lightSource.size();i++){
        app->removeLightSource(lightSource.at(i));
    }
    for(size_t i=0;i<lightPoint.size();i++)
        delete lightPoint.at(i);

    app->removeSkyBoxObject(skyboxObject);
    delete skyboxObject;

    for (size_t i =0 ;i<gltfModel.size();i++)
        for (size_t j =0 ;j<gltfModel.at(i).size();j++)
            app->destroyModel(gltfModel.at(i)[j]);

    app->removeBinds();
    for (size_t i=0 ;i<object3D.size();i++){
        delete object3D.at(i);
    }
}

void loadModels(VkApplication *app)
{
    size_t index = 0;

    gltfModel.resize(6);

    index = 0;
        gltfModel[0].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        app->createModel(gltfModel[0].at(index));
    index++;
        gltfModel[0].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        app->createModel(gltfModel[0].at(index));
    index++;
        gltfModel[0].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        app->createModel(gltfModel[0].at(index));
    index++;

    index = 0;
        gltfModel[1].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        app->createModel(gltfModel[1].at(index));
    index++;
        gltfModel[1].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        app->createModel(gltfModel[1].at(index));
    index++;
        gltfModel[1].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Bee.glb"));
        app->createModel(gltfModel[1].at(index));
    index++;

    index = 0;
        gltfModel[2].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Box.glb"));
        app->createModel(gltfModel[2].at(index));
    index++;

    index = 0;
        gltfModel[3].push_back(new struct gltfModel(ExternalPath + "model\\glTF\\Sponza\\Sponza.gltf"));
        app->createModel(gltfModel[3].at(index));
    index++;

    index = 0;
        gltfModel[4].push_back(new struct gltfModel(ExternalPath + "model\\glb\\Duck.glb"));
        app->createModel(gltfModel[4].at(index));
    index++;

    index = 0;
        gltfModel[5].push_back(new struct gltfModel(ExternalPath + "model\\glb\\RetroUFO.glb"));
        app->createModel(gltfModel[5].at(index));
    index++;
}

void createLight(VkApplication *app)
{
    glm::mat4x4 Proj = glm::perspective(glm::radians(spotAngle), 1.0f, 0.1f, 100.0f);
    Proj[1][1] *= -1;

    int index = 0;
    lightPoint.push_back(new light<pointLight>(lightSource));
    lightPoint.at(index)->setProjectionMatrix(Proj);
    lightPoint.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(lightPoint.at(index));

    for(int i=index;i<6;i++,index++){
        app->addLightSource(lightSource.at(index));
        //lightSource.at(i)->setScattering(true);
    }

    Proj = glm::perspective(glm::radians(spotAngle), 1.0f, 0.1f, 20.0f);
    Proj[1][1] *= -1;

    lightSource.push_back(new light<spotLight>(LIGHT_TEXTURE));
    lightSource.at(index)->setProjectionMatrix(Proj);
    //lightSource.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    groups.at(2)->addObject(lightSource.at(index));
    index++;
    app->addLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new light<spotLight>(LIGHT_TEXTURE));
    lightSource.at(index)->setProjectionMatrix(Proj);
    //lightSource.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    groups.at(3)->addObject(lightSource.at(index));
    index++;
    app->addLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new light<spotLight>(LIGHT_TEXTURE));
    lightSource.at(index)->setProjectionMatrix(Proj);
    //lightSource.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    groups.at(4)->addObject(lightSource.at(index));
    index++;
    app->addLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new light<spotLight>(LIGHT_TEXTURE));
    lightSource.at(index)->setProjectionMatrix(Proj);
    //lightSource.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    groups.at(5)->addObject(lightSource.at(index));
    index++;
    app->addLightSource(lightSource.at(lightSource.size()-1));

    for(int i=0;i<5;i++){
        lightSource.push_back(new light<spotLight>(LIGHT_TEXTURE));
        lightSource.at(index)->setProjectionMatrix(Proj);
        lightSource.at(index)->translate(glm::vec3(20.0f-10.0f*i,10.0f,3.0f));
        lightSource.at(index)->setScattering(true);
        index++;
    }

    for(int i=0;i<5;i++){
        lightSource.push_back(new light<spotLight>(LIGHT_TEXTURE));
        lightSource.at(index)->setProjectionMatrix(Proj);
        lightSource.at(index)->translate(glm::vec3(20.0f-10.0f*i,-10.0f,3.0f));
        lightSource.at(index)->setScattering(true);
        index++;
    }
}

void createObjects(VkApplication *app)
{
    uint32_t index=0;
    object3D.push_back( new object(gltfModel.at(0).size(),gltfModel.at(0).data()) );
    app->bindStencilObject(object3D.at(index),1.0f,glm::vec4(0.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->translate(glm::vec3(3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    index++;

    object3D.push_back( new object(gltfModel.at(1).size(),gltfModel.at(1).data()) );
    app->bindStencilObject(object3D.at(index),1.0f,glm::vec4(1.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->translate(glm::vec3(-3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    object3D.at(index)->animationTimer = 1.0f;
    object3D.at(index)->animationIndex = 1;
    index++;

    object3D.push_back( new object(gltfModel.at(4).size(),gltfModel.at(4).data()) );
    app->bindStencilObject(object3D.at(index),1.0f,glm::vec4(0.7f,0.5f,0.2f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    //object3D.at(index)->scale(glm::vec3(1.0f,1.0f,1.0f));
    object3D.at(index)->scale(glm::vec3(0.01f,0.01f,0.01f));
    object3D.at(index)->animationTimer = 0.0f;
    object3D.at(index)->animationIndex = 0;
    object *Duck = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(3).size(),gltfModel.at(3).data()) );
    app->bindBaseObject(object3D.at(index));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(3.0f,3.0f,3.0f));
    index++;

    object3D.push_back( new object(gltfModel.at(2).size(),gltfModel.at(2).data()) );
    app->bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    object *Box0 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    app->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO1 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    app->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO2 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    app->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,1.0f,0.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO3 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    app->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(0.0f,1.0f,1.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO4 = object3D.at(index);
    index++;

    groups.at(0)->translate(glm::vec3(0.0f,0.0f,5.0f));
    groups.at(0)->addObject(Box0);

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

void mouseEvent(VkApplication *app, GLFWwindow* window, float frameTime)
{
    static_cast<void>(frameTime);

    double x, y;

    int primitiveNumber = INT_FAST32_MAX;
    for(uint32_t i=0;i<app->getImageCount();i++){
        primitiveNumber = app->readStorageBuffer(i);
        if(primitiveNumber!=INT_FAST32_MAX)
            break;
    }

    glfwSetScrollCallback(window,&scrol);

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        double sensitivity = 0.001;
        glfwGetCursorPos(window,&x,&y);
        angx = sensitivity*(x - xMpos);
        angy = sensitivity*(y - yMpos);
        xMpos = x;
        yMpos = y;
        cameras->rotateX(angy,glm::vec3(1.0f,0.0f,0.0f));
        cameras->rotateY(angx,glm::vec3(0.0f,0.0f,-1.0f));
        app->resetUboWorld();

        for(uint32_t i=0;i<app->getImageCount();i++){
            app->updateStorageBuffer(i,glm::vec4(-1.0f+2.0f*xMpos/(WIDTH),-1.0f+2.0f*yMpos/(HEIGHT),0.0f,0.0f));
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
        app->resetUboLight();
        updateLightCone = false;
    }

    if(updateCamera){
        if(cameraAngle>0.0f){
            glm::mat4x4 proj = glm::perspective(glm::radians(cameraAngle), (float) WIDTH / (float) HEIGHT, 0.1f, 1000.0f);
            proj[1][1] *= -1.0f;
            cameras->setProjMatrix(proj);
        }
        app->resetUboWorld();
        updateCamera = false;
    }
}

void keyboardEvent(VkApplication *app, GLFWwindow* window, float frameTime)
{
    float sensitivity = 5.0f*frameTime;
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[0][2];
        float y = -sensitivity*cameras->getViewMatrix()[1][2];
        float z = -sensitivity*cameras->getViewMatrix()[2][2];
        cameras->translate(glm::vec3(x,y,z));
        app->resetUboWorld();
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[0][2];
        float y = sensitivity*cameras->getViewMatrix()[1][2];
        float z = sensitivity*cameras->getViewMatrix()[2][2];
        cameras->translate(glm::vec3(x,y,z));
        app->resetUboWorld();
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[0][0];
        float y = -sensitivity*cameras->getViewMatrix()[1][0];
        float z = -sensitivity*cameras->getViewMatrix()[2][0];
        cameras->translate(glm::vec3(x,y,z));
        app->resetUboWorld();
    }
    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[0][0];
        float y = sensitivity*cameras->getViewMatrix()[1][0];
        float z = sensitivity*cameras->getViewMatrix()[2][0];
        cameras->translate(glm::vec3(x,y,z));
        app->resetUboWorld();
    }
    if(glfwGetKey(window,GLFW_KEY_Z) == GLFW_PRESS)
    {
        float x = sensitivity*cameras->getViewMatrix()[0][1];
        float y = sensitivity*cameras->getViewMatrix()[1][1];
        float z = sensitivity*cameras->getViewMatrix()[2][1];
        cameras->translate(glm::vec3(x,y,z));
        app->resetUboWorld();
    }
    if(glfwGetKey(window,GLFW_KEY_X) == GLFW_PRESS)
    {
        float x = -sensitivity*cameras->getViewMatrix()[0][1];
        float y = -sensitivity*cameras->getViewMatrix()[1][1];
        float z = -sensitivity*cameras->getViewMatrix()[2][1];
        cameras->translate(glm::vec3(x,y,z));
        app->resetUboWorld();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_4) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(0.0f,0.0f,1.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_6) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(0.0f,0.0f,1.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_8) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(1.0f,0.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_5) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(1.0f,0.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_7) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(0.0f,1.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_9) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(0.0f,1.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(-1.0f,0.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(1.0f,0.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,1.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,-1.0f,0.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_ADD) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,0.0f,1.0f));
        app->resetUboWorld();
        app->resetUboLight();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,0.0f,-1.0f));
        app->resetUboWorld();
        app->resetUboLight();
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
        object3D[0]->setStencilEnable(!object3D[0]->getStencilEnable());
        object3D[1]->setStencilEnable(!object3D[1]->getStencilEnable());
        object3D[2]->setStencilEnable(!object3D[2]->getStencilEnable());
        app->resetCmdWorld();
    }
    backRStage = glfwGetKey(window,GLFW_KEY_R);
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
        app->bindBaseObject(object3D.at(index));
        object3D.at(index)->translate(cameras->getTranslate());
        object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
        app->resetCmdWorld();
        app->resetCmdLight();
        app->resetUboWorld();
        app->resetUboLight();
    }
    backNStage = glfwGetKey(window,GLFW_KEY_N);
    if(backBStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_B) == 0)
    {
        app->deviceWaitIdle();
        if(object3D.size()>0){
            size_t index = object3D.size()-1;
            if(app->removeBaseObject(object3D[index]))
            {
                delete object3D[index];
                object3D.erase(object3D.begin()+index);
                app->resetCmdWorld();
                app->resetCmdLight();
                app->resetUboWorld();
            }
        }
    }
    backBStage = glfwGetKey(window,GLFW_KEY_B);
    if(backGStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_G) == 0)
    {
        if(lightPointer<20){
            app->addLightSource(lightSource.at(lightPointer));
            lightPointer++;
        }
        app->resetCmdWorld();
        app->resetCmdLight();
        app->resetUboLight();
    }
    backGStage = glfwGetKey(window,GLFW_KEY_G);
    if(backHStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_H) == 0)
    {
        app->deviceWaitIdle();
        if(lightPointer>0){
            lightPointer--;
            app->removeLightSource(lightSource.at(lightPointer));
        }
        app->resetCmdWorld();
        app->resetCmdLight();
        app->resetUboLight();

    }
    backHStage = glfwGetKey(window,GLFW_KEY_H);

    if(glfwGetKey(window,GLFW_KEY_KP_0) == GLFW_PRESS)
    {
        minAmbientFactor -= 0.1f*sensitivity;
        app->setMinAmbientFactor(minAmbientFactor);
        app->resetCmdWorld();
    }
    if(glfwGetKey(window,GLFW_KEY_KP_2) == GLFW_PRESS)
    {
        minAmbientFactor += 0.1f*sensitivity;
        app->setMinAmbientFactor(minAmbientFactor);
        app->resetCmdWorld();
    }
}

void scrol(GLFWwindow *window, double xoffset, double yoffset)
{
    static_cast<void>(window);

    spotAngle -= yoffset;
    updateLightCone = true;

    cameraAngle -= xoffset;
    updateCamera = true;
}

#endif
