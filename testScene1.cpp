#ifdef TestScene1

#include "core/vulkanCore.h"
#include "core/texture.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/transformational/object.h"
#include "core/transformational/group.h"
#include "core/transformational/camera.h"
#include "libs/stb-master/stb_image.h"

#include <chrono>
#include <stdexcept>        // предотвращения ошибок
#include <cstdlib>          // заголовок для использования макросов EXIT_SUCCESS и EXIT_FAILURE
#include <sstream>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 800;

std::string ZERO_TEXTURE        = ExternalPath + "texture\\0.png";
std::string ZERO_TEXTURE_WHITE  = ExternalPath + "texture\\1.png";
std::string LIGHT_TEXTURE  = ExternalPath + "texture\\light1.jpg";

std::vector<std::string> SKYBOX = {
    ExternalPath+"texture\\skybox\\left.jpg",
    ExternalPath+"texture\\skybox\\right.jpg",
    ExternalPath+"texture\\skybox\\front.jpg",
    ExternalPath+"texture\\skybox\\back.jpg",
    ExternalPath+"texture\\skybox\\top.jpg",
    ExternalPath+"texture\\skybox\\bottom.jpg"
};

bool framebufferResized = false;

float frameTime;
float fps = 60.0f;
bool  animate = true;
bool  fpsLock = false;

double   xMpos, yMpos;
double   angx=0.0, angy=0.0;

uint32_t controledGroup = 0;

bool     backRStage = 0;
bool     backTStage = 0;
bool     backYStage = 0;
bool     backNStage = 0;
bool     backOStage = 0;
bool     backIStage = 0;

void scrol(GLFWwindow* window, double xoffset, double yoffset);
void mouseEvent(VkApplication* app, GLFWwindow* window, float frameTime, std::vector<object*>& object3D, std::vector<group*>& groups, std::vector<gltfModel*>& gltfModel, texture *emptyTexture, camera* cameras);
void keyboardEvent(VkApplication* app, GLFWwindow* window, float frameTime, std::vector<object*>& object3D, std::vector<group*>& groups, std::vector<gltfModel*>& gltfModel, texture *emptyTexture, camera* cameras);

void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void initializeWindow(GLFWwindow* &window);
void recreateSwapChain(VkApplication* app, GLFWwindow* window);

void loadModels(VkApplication* app, std::vector<gltfModel *>& gltfModel);
void createLight(VkApplication* app, std::vector<light<spotLight>*>& lightSource, std::vector<light<pointLight>*>& lightPoint, std::vector<group*>& groups, texture* lightTex1);
void createObjects(VkApplication* app, std::vector<gltfModel*>& gltfModel, std::vector<object*>& object3D, std::vector<group*>& groups, texture* emptyTexture, texture* emptyTextureW);

int testScene1()
{
    try
    {
        std::vector<gltfModel           *>          gltfModel;
        std::vector<object              *>          object3D;
        std::vector<light<spotLight>    *>          lightSource;
        std::vector<light<pointLight>   *>          lightPoint;
        std::vector<group               *>          groups;

        groups.push_back(new group);
        groups.push_back(new group);
        groups.push_back(new group);
        groups.push_back(new group);
        groups.push_back(new group);
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

        loadModels(&app,gltfModel);

        camera *cameras = new camera;
            cameras->translate(glm::vec3(0.0f,0.0f,10.0f));
        app.addCamera(cameras);

        texture *emptyTexture = new texture(&app,ZERO_TEXTURE);
            emptyTexture->createTextureImage();
            emptyTexture->createTextureImageView();
            emptyTexture->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        texture *emptyTextureW = new texture(&app,ZERO_TEXTURE_WHITE);
            emptyTextureW->createTextureImage();
            emptyTextureW->createTextureImageView();
            emptyTextureW->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        app.getGraphics().setEmptyTexture(emptyTexture);

        cubeTexture *skybox = new cubeTexture(&app,SKYBOX);
            skybox->setMipLevel(0.0f);
            skybox->createTextureImage();
            skybox->createTextureImageView();
            skybox->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        object *skyboxObject = new object(&app,{gltfModel.at(2),nullptr});
            skyboxObject->scale(glm::vec3(200.0f,200.0f,200.0f));
        app.getGraphics().bindSkyBoxObject(skyboxObject);
        app.getGraphics().setSkyboxTexture(skybox);

        texture *lightTex1 = new texture(&app,LIGHT_TEXTURE);
            lightTex1->createTextureImage();
            lightTex1->createTextureImageView();
            lightTex1->createTextureSampler({VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});

        app.createGraphics(window);

        createLight(&app,lightSource,lightPoint,groups, lightTex1);
        createObjects(&app,gltfModel,object3D,groups,emptyTexture,emptyTextureW);

        app.updateDescriptorSets();
        app.createCommandBuffers();
        app.createSyncObjects();

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

                if(animate)
                    for(size_t j=0;j<object3D.size();j++)
                    {
                        object3D[j]->animationTimer += frameTime;
                        object3D[j]->updateAnimation();
                    }

                glfwPollEvents();
                mouseEvent(&app,window,frameTime,object3D,groups,gltfModel,emptyTexture,cameras);
                keyboardEvent(&app,window,frameTime,object3D,groups,gltfModel,emptyTexture,cameras);
                VkResult result = app.drawFrame();

                if (result == VK_ERROR_OUT_OF_DATE_KHR)                         recreateSwapChain(&app,window);
                else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)   throw std::runtime_error("failed to acquire swap chain image!");

                if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
                    framebufferResized = false;
                    recreateSwapChain(&app,window);
                }else if (result != VK_SUCCESS) throw std::runtime_error("failed to present swap chain image!");
            }

            vkDeviceWaitIdle(app.getDevice());

        emptyTexture->destroy(); delete emptyTexture;
        emptyTextureW->destroy(); delete emptyTextureW;

        lightTex1->destroy(); delete lightTex1;

        app.removeLightSources();
        for(size_t i=0;i<lightSource.size();i++)
            lightSource.at(i)->cleanup();
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
    vkDeviceWaitIdle(app->getDevice());

    app->destroyGraphics();
    app->freeCommandBuffers();

    app->createGraphics(window);
    app->updateDescriptorSets();
    app->createCommandBuffers();
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
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\glb\\Bee.glb",app,1.0f);
    index++;

        gltfModel.push_back(new struct gltfModel);
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\glb\\Bee.glb",app,1.0f);
    index++;

        gltfModel.push_back(new struct gltfModel);
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\glb\\Box.glb",app,1.0f);
    index++;

        gltfModel.push_back(new struct gltfModel);
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\glTF\\Sponza\\Sponza.gltf",app,1.0f);
    index++;

        gltfModel.push_back(new struct gltfModel);
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\glb\\Robot.glb",app,1.0f);
    index++;

        gltfModel.push_back(new struct gltfModel);
        gltfModel.at(index)->loadFromFile(ExternalPath + "model\\glb\\RetroUFO.glb",app,1.0f);
    index++;
}

void createLight(VkApplication *app, std::vector<light<spotLight>*>& lightSource, std::vector<light<pointLight>*>& lightPoint, std::vector<group*>& groups, texture* lightTex1)
{
    int index = 0;
    lightPoint.push_back(new light<pointLight>(app,lightSource));
    lightPoint.at(0)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(lightPoint.at(0));
    index +=6;

    glm::mat4x4 Proj;

    lightSource.push_back(new light<spotLight>(app));
        Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
        Proj[1][1] *= -1;
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->setLightNumber(index);
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,0.0f,1.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    lightSource.at(index)->setTexture(lightTex1);
    groups.at(2)->addObject(lightSource.at(index));
    index++;
    app->addlightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new light<spotLight>(app));
        Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
        Proj[1][1] *= -1;
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->setLightNumber(index);
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,0.0f,0.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    lightSource.at(index)->setTexture(lightTex1);
    groups.at(3)->addObject(lightSource.at(index));
    index++;
    app->addlightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new light<spotLight>(app));
        Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
        Proj[1][1] *= -1;
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->setLightNumber(index);
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,1.0f,0.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    lightSource.at(index)->setTexture(lightTex1);
    groups.at(4)->addObject(lightSource.at(index));
    index++;
    app->addlightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new light<spotLight>(app));
        Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
        Proj[1][1] *= -1;
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->setLightNumber(index);
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,1.0f,1.0f,0.0f));
    lightSource.at(index)->setScattering(true);
    lightSource.at(index)->setTexture(lightTex1);
    groups.at(5)->addObject(lightSource.at(index));
    index++;
    app->addlightSource(lightSource.at(lightSource.size()-1));
}

void createObjects(VkApplication *app, std::vector<gltfModel*>& gltfModel, std::vector<object*>& object3D, std::vector<group*>& groups, texture *emptyTexture, texture *emptyTextureW)
{
    uint32_t index=0;
    object3D.push_back( new object(app,{gltfModel.at(0),emptyTexture}) );
    app->getGraphics().bindStencilObject(object3D.at(index),1.0f,glm::vec4(0.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->translate(glm::vec3(3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    index++;

    object3D.push_back( new object(app,{gltfModel.at(1),emptyTexture}) );
    app->getGraphics().bindStencilObject(object3D.at(index),1.0f,glm::vec4(1.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->translate(glm::vec3(-3.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    object3D.at(index)->animationTimer = 1.0f;
    object3D.at(index)->animationIndex = 1;
    index++;

    object3D.push_back( new object(app,{gltfModel.at(4),emptyTexture}) );
    app->getGraphics().bindStencilObject(object3D.at(index),1.0f,glm::vec4(0.7f,0.5f,0.2f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(15.0f,15.0f,15.0f));
    object3D.at(index)->animationTimer = 0.0f;
    object3D.at(index)->animationIndex = 0;
    object *Duck = object3D.at(index);
    index++;

    object3D.push_back( new object(app,{gltfModel.at(3),emptyTexture}) );
    app->getGraphics().bindBaseObject(object3D.at(index));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(3.0f,3.0f,3.0f));
    index++;

    object3D.push_back( new object(app,{gltfModel.at(2),emptyTextureW}) );
    app->getGraphics().bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    object *Box0 = object3D.at(index);
    index++;

    object3D.push_back( new object(app,{gltfModel.at(2),emptyTextureW}) );
    app->getGraphics().bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));
    object *Box1 = object3D.at(index);
    index++;

    object3D.push_back( new object(app,{gltfModel.at(2),emptyTextureW}) );
    app->getGraphics().bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));
    object *Box2 = object3D.at(index);
    index++;

    object3D.push_back( new object(app,{gltfModel.at(2),emptyTextureW}) );
    app->getGraphics().bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,1.0f,0.0f,1.0f));
    object *Box3 = object3D.at(index);
    index++;

    object3D.push_back( new object(app,{gltfModel.at(2),emptyTextureW}) );
    app->getGraphics().bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(0.0f,1.0f,1.0f,1.0f));
    object *Box4 = object3D.at(index);
    index++;

    groups.at(0)->translate(glm::vec3(0.0f,0.0f,5.0f));
    groups.at(0)->addObject(Box0);

    groups.at(1)->addObject(Duck);

    groups.at(2)->translate(glm::vec3(5.0f,0.0f,5.0f));
    groups.at(2)->addObject(Box1);

    groups.at(3)->translate(glm::vec3(-5.0f,0.0f,5.0f));
    groups.at(3)->addObject(Box2);

    groups.at(4)->translate(glm::vec3(10.0f,0.0f,5.0f));
    groups.at(4)->addObject(Box3);

    groups.at(5)->translate(glm::vec3(-10.0f,0.0f,5.0f));
    groups.at(5)->addObject(Box4);
}

void mouseEvent(VkApplication *app, GLFWwindow* window, float frameTime, std::vector<object*>& object3D, std::vector<group*>& groups, std::vector<gltfModel*>& gltfModel, texture *emptyTexture, camera* cameras)
{
    static_cast<void>(frameTime);
    static_cast<void>(gltfModel);
    static_cast<void>(object3D);
    static_cast<void>(groups);
    static_cast<void>(emptyTexture);

    double x, y;
    int button = glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT);
    glfwSetScrollCallback(window,&scrol);
    if(button == GLFW_PRESS)
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
    }
    else
    {
        glfwGetCursorPos(window,&xMpos,&yMpos);
    }
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
        app->getGraphics().setStencilObject(object3D[0]);
        app->getGraphics().setStencilObject(object3D[1]);
        app->getGraphics().setStencilObject(object3D.at(2));
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
        object3D.push_back( new object(app,{gltfModel.at(5),emptyTexture}) );
        app->getGraphics().bindStencilObject(object3D.at(index),1.0f,glm::vec4(0.0f,0.5f,0.8f,1.0f));
        object3D.at(index)->translate(cameras->getTranslate());
        object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
        app->resetCmdWorld();
        app->resetCmdLight();
        app->resetUboWorld();
        app->resetUboLight();
    }
    backNStage = glfwGetKey(window,GLFW_KEY_N);
}

void scrol(GLFWwindow *window, double xoffset, double yoffset)
{
    static_cast<void>(window);
    static_cast<void>(xoffset);
    static_cast<void>(yoffset);
}

#endif
