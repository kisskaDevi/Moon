#ifndef TESTPOS_H
#define TESTPOS_H

#include <filesystem>
#include <vector>
#include <unordered_map>

#include "scene.h"
#include "matrix.h"

#include "controller.h"

class deferredGraphics;
class graphicsManager;
class model;
class baseObject;
class skyboxObject;
class baseCamera;
class spotLight;
class isotropicLight;

class testPos : public scene
{
private:
    std::filesystem::path   ExternalPath;
    vector<uint32_t,2>      extent{0};
    vector<double,2>        mousePos{0.0};
    float                   globalTime{0.0f};
    float                   minAmbientFactor{0.28f};
    vector<float,3>         maxSize{0.0f};

    GLFWwindow*         window{nullptr};
    graphicsManager*    app{nullptr};
    baseObject*         selectedObject{nullptr};
    controller*         mouse{nullptr};
    controller*         board{nullptr};

    std::unordered_map<std::string, baseCamera*>         cameras;
    std::unordered_map<std::string, deferredGraphics*>   graphics;
    std::unordered_map<std::string, spotLight*>          lights;

    std::unordered_map<std::string, model*>              models;
    std::unordered_map<matrix<float,4,4>*, std::string>  cameraNames;
    std::unordered_map<matrix<float,4,4>*, baseObject*>  cameraObjects;
    std::unordered_map<std::string, baseObject*>         staticObjects;
    std::unordered_map<std::string, skyboxObject*>       skyboxObjects;
    std::unordered_map<std::string, isotropicLight*>     lightPoints;
    std::vector<spotLight*>                              lightSources;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    testPos(graphicsManager *app, GLFWwindow* window, const std::filesystem::path& ExternalPath);
    void create(uint32_t WIDTH, uint32_t HEIGHT) override;
    void resize(uint32_t WIDTH, uint32_t HEIGHT) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
    void destroy() override;
};

#endif // TESTPOS_H
