#ifndef TESTSCENE_H
#define TESTSCENE_H

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <glfw3.h>

#include "scene.h"
#include "dualQuaternion.h"

class deferredGraphics;
class model;
class graphicsManager;
class spotLight;
class isotropicLight;
class baseObject;
class group;
class baseCamera;
class skyboxObject;
class plyModel;
class transformational;

class controler
{
private:
    std::unordered_map<uint32_t, bool> keysMap;
    GLFWwindow* window{nullptr};
    int (*glfwGetFunction)(GLFWwindow*,int){nullptr};

public:
    float sensitivity{0.1f};

public:
    controler() = default;
    controler(GLFWwindow* window, int (*glfwGetFunction)(GLFWwindow*,int)) : window(window), glfwGetFunction(glfwGetFunction){}

    bool pressed(uint32_t key){
        bool res = glfwGetFunction(window,key) == GLFW_PRESS;
        return res;
    }

    bool released(uint32_t key){
        bool res = keysMap.count(key) > 0 && keysMap[key] == GLFW_PRESS && glfwGetFunction(window,key) == GLFW_RELEASE;
        keysMap[key] = glfwGetFunction(window,key);
        return res;
    }
};

class testScene : public scene
{
private:
    std::filesystem::path   ExternalPath;
    vector<uint32_t,2>      extent{0};
    vector<double,2>        mousePos{0.0};
    float                   globalTime = 0.0f, timeScale = 1.0f;
    float                   minAmbientFactor = 0.05f;
    int                     ufoCounter = 0;

    transformational*                                   controledObject{nullptr};
    GLFWwindow*                                         window{nullptr};
    graphicsManager*                                    app{nullptr};
    controler*                                          mouse{nullptr};
    controler*                                          board{nullptr};

    std::unordered_map<std::string, baseCamera*>        cameras;
    std::unordered_map<std::string, deferredGraphics*>  graphics;

    std::unordered_map<std::string, model*>             models;
    std::unordered_map<std::string, baseObject*>        objects;
    std::unordered_map<std::string, baseObject*>        staticObjects;
    std::unordered_map<std::string, skyboxObject*>      skyboxObjects;
    std::unordered_map<std::string, group*>             groups;
    std::unordered_map<std::string, isotropicLight*>    lightPoints;
    std::vector<spotLight*>                             lightSources;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    testScene(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath);
    ~testScene();

    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT) override;
    void destroy() override;
};

#endif // TESTSCENE_H
