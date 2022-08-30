#ifdef TestScene2

uint32_t controledGroup = 0;

void scrol(GLFWwindow* window, double xoffset, double yoffset);
void mouseEvent(VkApplication* app, GLFWwindow* window, float frameTime, std::vector<object*>& object3D, std::vector<group*>& groups, std::vector<gltfModel*>& gltfModel, texture *emptyTexture, camera* cameras);
void keyboardEvent(VkApplication* app, GLFWwindow* window, float frameTime, std::vector<object*>& object3D, std::vector<group*>& groups, std::vector<gltfModel*>& gltfModel, texture *emptyTexture, camera* cameras);

void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void initializeWindow(GLFWwindow* &window);
void recreateSwapChain(VkApplication* app, GLFWwindow* window);

void loadModels(VkApplication* app, std::vector<gltfModel *>& gltfModel);
void createLight(VkApplication* app, std::vector<light<spotLight>*>& lightSource, std::vector<light<pointLight>*>& lightPoint, std::vector<group*>& groups);
void createObjects(VkApplication* app, std::vector<gltfModel*>& gltfModel, std::vector<object*>& object3D, std::vector<group*>& groups, texture* emptyTexture, texture* emptyTextureW);

int testScene2()
{
    try
    {
        std::vector<gltfModel           *>          gltfModel;
        std::vector<object              *>          object3D;
        std::vector<light<spotLight>    *>          lightSource;
        std::vector<light<pointLight>   *>          lightPoint;
        std::vector<group               *>          groups;

        groups.push_back(new group);

        GLFWwindow* window;
            initializeWindow(window);

        VkApplication app;
        app.createInstance();
        app.setupDebugMessenger();
        app.createSurface(window);
        app.pickPhysicalDevice();
        app.createLogicalDevice();
        app.checkSwapChainSupport();
        app.createCommandPool();

        camera *cameras = new camera;
            cameras->translate(glm::vec3(0.0f,0.0f,10.0f));
            glm::mat4x4 proj = glm::perspective(glm::radians(45.0f), (float) WIDTH / (float) HEIGHT, 0.1f, 1000.0f);
            proj[1][1] *= -1.0f;
            cameras->setProjMatrix(proj);
        app.getGraphics().setCameraObject(cameras);

        texture *emptyTexture = new texture(&app,ZERO_TEXTURE);
            emptyTexture->createTextureImage();
            emptyTexture->createTextureImageView();
            emptyTexture->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        texture *emptyTextureW = new texture(&app,ZERO_TEXTURE_WHITE);
            emptyTextureW->createTextureImage();
            emptyTextureW->createTextureImageView();
            emptyTextureW->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        app.getGraphics().setEmptyTexture(emptyTexture);

        app.createGraphics(window);
        loadModels(&app,gltfModel);

        cubeTexture *skybox = new cubeTexture(&app,SKYBOX);
            skybox->setMipLevel(0.0f);
            skybox->createTextureImage();
            skybox->createTextureImageView();
            skybox->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        object *skyboxObject = new object(&app,{gltfModel.at(1),nullptr});
            skyboxObject->scale(glm::vec3(200.0f,200.0f,200.0f));
        app.getGraphics().bindSkyBoxObject(skyboxObject,skybox);

        createLight(&app,lightSource,lightPoint,groups);
        createObjects(&app,gltfModel,object3D,groups,emptyTexture,emptyTextureW);

        app.updateDescriptorSets();
        app.createCommandBuffers();
        app.createSyncObjects();

        app.resetUboWorld();
        app.resetUboLight();

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
                mouseEvent(&app,window,frameTime,object3D,groups,gltfModel,emptyTexture,cameras);
                keyboardEvent(&app,window,frameTime,object3D,groups,gltfModel,emptyTexture,cameras);

                VkResult result = app.drawFrame();

                if (result == VK_ERROR_OUT_OF_DATE_KHR)                         recreateSwapChain(&app,window);
                else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)   throw std::runtime_error("failed to acquire swap chain image!");

                if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
                    framebufferResized = false;
                    recreateSwapChain(&app,window);
                    glm::mat4x4 proj = glm::perspective(glm::radians(45.0f), (float) WIDTH / (float) HEIGHT, 0.1f, 1000.0f);
                    proj[1][1] *= -1.0f;
                    cameras->setProjMatrix(proj);
                }else if (result != VK_SUCCESS) throw std::runtime_error("failed to present swap chain image!");
            }

            vkDeviceWaitIdle(app.getDevice());

        emptyTexture->destroy(); delete emptyTexture;
        emptyTextureW->destroy(); delete emptyTextureW;

        app.removeLightSources();
        for(size_t i=0;i<lightSource.size();i++){
            lightSource.at(i)->destroyBuffer();
            lightSource.at(i)->cleanup();
        }
        for(size_t i=0;i<lightPoint.size();i++)
            delete lightPoint.at(i);

        skyboxObject->destroyUniformBuffers(); delete skyboxObject;
        skybox->destroy(); delete skybox;

        app.getGraphics().removeBinds();
        for (size_t i=0 ;i<object3D.size();i++){
            object3D.at(i)->destroyUniformBuffers();
            delete object3D.at(i);
        }

        for (size_t i =0 ;i<gltfModel.size();i++)
            gltfModel.at(i)->destroy(gltfModel.at(i)->app->getDevice());

        app.cleanup();

        glfwDestroyWindow(window);
        glfwTerminate();

    } catch (const std::exception& e) {
        std::cerr<<e.what()<<std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void recreateSwapChain(VkApplication *app, GLFWwindow* window)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }
    WIDTH = width;
    HEIGHT = height;
    vkDeviceWaitIdle(app->getDevice());

    app->destroyGraphics();
    app->freeCommandBuffers();

    app->checkSwapChainSupport();
    app->createGraphics(window);
    app->updateDescriptorSets();
    app->createCommandBuffers();

    app->resetUboWorld();
    app->resetUboLight();
}

void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    static_cast<void>(width);
    static_cast<void>(height);
    static_cast<void>(window);
    framebufferResized = true;
}
void initializeWindow(GLFWwindow* &window)
{
    glfwInit();                                                             //инициализация библиотеки GLFW
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);                           //указывает не создавать контекст OpenGL (GLFW изначально был разработан для создания контекста OpenGL)

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);   //инициализация собственного окна
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    int width,height,comp;
    std::string filename = ExternalPath + "texture\\icon.png";
    stbi_uc* img = stbi_load(filename.c_str(), &width, &height, &comp, 0);
    GLFWimage images{width,height,img};
    glfwSetWindowIcon(window,1,&images);
}

void loadModels(VkApplication *app, std::vector<gltfModel *>& gltfModel)
{
    size_t index = 0;

        gltfModel.push_back(new struct gltfModel);
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\hw1.gltf",app,1.0f);
        app->getGraphics().createModel(gltfModel.at(index));
    index++;

        gltfModel.push_back(new struct gltfModel);
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\glTF\\box\\Cube.gltf",app,1.0f);
        app->getGraphics().createModel(gltfModel.at(index));
    index++;

}

void createLight(VkApplication *app, std::vector<light<spotLight>*>& lightSource, std::vector<light<pointLight>*>& lightPoint, std::vector<group*>& groups)
{
//    lightPoint.push_back(new light<pointLight>(app,lightSource));
//    lightPoint.at(0)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
//    groups.at(0)->addObject(lightPoint.at(0));

    uint32_t index = 0;
    glm::mat4x4 Proj;

    lightSource.push_back(new light<spotLight>(app));
        Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 20.0f);
        Proj[1][1] *= -1;
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    lightSource.at(index)->setScattering(true);
    groups.at(0)->addObject(lightSource.at(index));
    app->addLightSource(lightSource.at(lightSource.size()-1));
    index++;
}

void createObjects(VkApplication *app, std::vector<gltfModel*>& gltfModel, std::vector<object*>& object3D, std::vector<group*>& groups, texture *emptyTexture, texture *emptyTextureW)
{
    uint32_t index=0;

    object3D.push_back( new object(app,{gltfModel.at(1),emptyTextureW}) );
    app->getGraphics().bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));
    object *Box = object3D.at(index);
    index++;

    object3D.push_back( new object(app,{gltfModel.at(0),emptyTextureW}) );
    app->getGraphics().bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.01f,0.01f,0.01f));
    index++;

    groups.at(0)->translate(glm::vec3(0.0f,0.0f,10.0f));
    groups.at(0)->addObject(Box);
}

void mouseEvent(VkApplication *app, GLFWwindow* window, float frameTime, std::vector<object*>& object3D, std::vector<group*>& groups, std::vector<gltfModel*>& gltfModel, texture *emptyTexture, camera* cameras)
{
    static_cast<void>(frameTime);
    static_cast<void>(gltfModel);
    static_cast<void>(object3D);
    static_cast<void>(groups);
    static_cast<void>(emptyTexture);

    double x, y;
    double sensitivity = 0.001;
    int n = INT_FAST32_MAX;
    for(uint32_t i=0;i<3;i++){
        n = app->getGraphics().readStorageBuffer(i);
        if(n!=INT_FAST32_MAX)
            break;
    }

    glfwSetScrollCallback(window,&scrol);
    if(mouse1Stage == GLFW_PRESS && glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == 1)
    {
        glfwGetCursorPos(window,&x,&y);
        angx = sensitivity*(x - xMpos);
        angy = sensitivity*(y - yMpos);
        xMpos = x;
        yMpos = y;
        cameras->rotateX(angy,glm::vec3(1.0f,0.0f,0.0f));
        cameras->rotateY(angx,glm::vec3(0.0f,0.0f,-1.0f));
        app->resetUboWorld();
        app->getGraphics().updateStorageBuffer(0,glm::vec4(-1.0f+2.0f*xMpos/(WIDTH),-1.0f+2.0f*yMpos/(HEIGHT),0.0f,0.0f));
        app->getGraphics().updateStorageBuffer(1,glm::vec4(-1.0f+2.0f*xMpos/(WIDTH),-1.0f+2.0f*yMpos/(HEIGHT),0.0f,0.0f));
        app->getGraphics().updateStorageBuffer(2,glm::vec4(-1.0f+2.0f*xMpos/(WIDTH),-1.0f+2.0f*yMpos/(HEIGHT),0.0f,0.0f));
    }
    else if(mouse1Stage == GLFW_PRESS && glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == 0)
    {

    }
    else
    {
        glfwGetCursorPos(window,&xMpos,&yMpos);
    }
    mouse1Stage = glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT);
}

void keyboardEvent(VkApplication *app, GLFWwindow* window, float frameTime, std::vector<object*>& object3D, std::vector<group*>& groups, std::vector<gltfModel*>& gltfModel, texture *emptyTexture,camera* cameras)
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
    if(spaceStage == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_SPACE) == 0)
    {

    }
    spaceStage = glfwGetKey(window,GLFW_KEY_SPACE);
    if(glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window,GLFW_TRUE);
    }
}

void scrol(GLFWwindow *window, double xoffset, double yoffset)
{
    static_cast<void>(window);
    static_cast<void>(xoffset);
    static_cast<void>(yoffset);
}

#endif
