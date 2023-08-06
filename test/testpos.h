#ifndef TESTPOS_H
#define TESTPOS_H

#include <filesystem>
#include <vector>
#include <unordered_map>

#include "scene.h"
#include "matrix.h"

struct GLFWwindow;
class deferredGraphics;
class graphicsManager;
class model;
class baseObject;
class skyboxObject;
class baseCamera;
class spotLight;
class isotropicLight;

void scrol(GLFWwindow* window, double xoffset, double yoffset);

class testPos : public scene
{
private:
    uint32_t    WIDTH{0}, HEIGHT{0};
    double      xMpos{0.0}, yMpos{0.0};
    float       globalTime = 0.0f;
    float       minAmbientFactor = 0.28f;
    vector<float,3> maxSize{0.0f};
    std::filesystem::path ExternalPath;

    bool        mouse1Stage = 0;

    GLFWwindow*         window;
    graphicsManager*    app;
    baseObject*         selectedObject{nullptr};

    deferredGraphics*   globalSpaceView{nullptr};
    deferredGraphics*   localView{nullptr};

    baseCamera*         globalCamera{nullptr};
    baseCamera*         localCamera{nullptr};

    std::unordered_map<std::string, model*>              models;
    std::unordered_map<matrix<float,4,4>*, baseObject*>  cameraObject3D;
    std::unordered_map<std::string, baseObject*>         staticObject3D;
    std::vector<skyboxObject            *>               skyboxObjects;
    std::vector<spotLight               *>               lightSources;
    std::vector<isotropicLight          *>               lightPoints;

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
    void updateFrame(uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT) override;
    void destroy() override;
};

#endif // TESTPOS_H
