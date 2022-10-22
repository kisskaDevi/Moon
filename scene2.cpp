#include "scene2.h"
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>
#include "core/graphics/deferredGraphics/deferredgraphicsinterface.h"

scene2::scene2(graphicsManager *app, deferredGraphicsInterface* graphics, std::string ExternalPath)
{
    this->app = app;
    this->graphics = graphics;
    this->ExternalPath = ExternalPath;
}

void scene2::createScene(uint32_t WIDTH, uint32_t HEIGHT)
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

    cameras = new camera;
        cameras->translate(glm::vec3(0.0f,0.0f,10.0f));
        glm::mat4x4 proj = glm::perspective(glm::radians(45.0f), (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
        proj[1][1] *= -1.0f;
        cameras->setProjMatrix(proj);
    graphics->setCameraObject(cameras);

    skyboxObject = new object(1,nullptr);
        skyboxObject->scale(glm::vec3(200.0f,200.0f,200.0f));
    graphics->bindSkyBoxObject(skyboxObject, SKYBOX);

    loadModels();
    createLight();
    createObjects();
}

void scene2::updateFrame(GLFWwindow* window, uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT)
{
    this->WIDTH = WIDTH;
    this->HEIGHT = HEIGHT;

    glm::mat4x4 proj = glm::perspective(glm::radians(45.0f), (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
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

void scene2::destroyScene()
{
    for(size_t i=0;i<lightSource.size();i++){
        graphics->removeLightSource(lightSource.at(i));
    }
    for(size_t i=0;i<lightPoint.size();i++)
        delete lightPoint.at(i);

    graphics->removeSkyBoxObject(skyboxObject);
    delete skyboxObject;

    for (size_t i =0 ;i<gltfModel.size();i++)
        for (size_t j =0 ;j<gltfModel.at(i).size();j++)
            graphics->destroyModel(gltfModel.at(i)[j]);

    graphics->removeBinds();
    for (size_t i=0 ;i<object3D.size();i++){
        delete object3D.at(i);
    }

    delete cameras;
}

void scene2::loadModels()
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
        gltfModel[4].push_back(new struct gltfModel(ExternalPath + "model\\gltf\\kerosene_lamp\\scene.gltf"));
        graphics->createModel(gltfModel[4].at(index));
    index++;

    index = 0;
        gltfModel[5].push_back(new struct gltfModel(ExternalPath + "model\\glb\\RetroUFO.glb"));
        graphics->createModel(gltfModel[5].at(index));
    index++;
}

void scene2::createLight()
{
    std::string LIGHT_TEXTURE0  = ExternalPath + "texture\\icon.PNG";
    std::string LIGHT_TEXTURE1  = ExternalPath + "texture\\light1.jpg";
    std::string LIGHT_TEXTURE2  = ExternalPath + "texture\\light2.jpg";
    std::string LIGHT_TEXTURE3  = ExternalPath + "texture\\light3.jpg";

    glm::mat4x4 Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 100.0f);
    Proj[1][1] *= -1;

    int index = 0;
    lightPoint.push_back(new pointLight(lightSource));
    lightPoint.at(index)->setProjectionMatrix(Proj);
    lightPoint.at(index)->setLightColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    groups.at(0)->addObject(lightPoint.at(index));

    for(int i=index;i<6;i++,index++){
        graphics->addLightSource(lightSource.at(index));
    }

    Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 20.0f);
    Proj[1][1] *= -1;

    lightSource.push_back(new spotLight(LIGHT_TEXTURE0));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(2)->addObject(lightSource.at(index));
    index++;
    graphics->addLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new spotLight(LIGHT_TEXTURE1));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(3)->addObject(lightSource.at(index));
    index++;
    graphics->addLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new spotLight(LIGHT_TEXTURE2));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(4)->addObject(lightSource.at(index));
    index++;
    graphics->addLightSource(lightSource.at(lightSource.size()-1));

    lightSource.push_back(new spotLight(LIGHT_TEXTURE3));
    lightSource.at(index)->setProjectionMatrix(Proj);
    lightSource.at(index)->setScattering(true);
    groups.at(5)->addObject(lightSource.at(index));
    index++;
    graphics->addLightSource(lightSource.at(lightSource.size()-1));
}

void scene2::createObjects()
{
    uint32_t index=0;
    object3D.push_back( new object(gltfModel.at(0).size(),gltfModel.at(0).data()) );
    graphics->bindStencilObject(object3D.at(index),0.05f,glm::vec4(0.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->translate(glm::vec3(5.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    index++;

    object3D.push_back( new object(gltfModel.at(1).size(),gltfModel.at(1).data()) );
    graphics->bindStencilObject(object3D.at(index),0.05f,glm::vec4(1.0f,0.5f,0.8f,1.0f));
    object3D.at(index)->translate(glm::vec3(-5.0f,0.0f,0.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(0.2f,0.2f,0.2f));
    object3D.at(index)->animationTimer = 1.0f;
    object3D.at(index)->animationIndex = 1;
    index++;

    object3D.push_back( new object(gltfModel.at(3).size(),gltfModel.at(3).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object3D.at(index)->scale(glm::vec3(3.0f,3.0f,3.0f));
    index++;

    object3D.push_back( new object(gltfModel.at(2).size(),gltfModel.at(2).data()) );
    graphics->bindBloomObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,1.0f,1.0f,1.0f));
    object *Box0 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO1 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO2 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(1.0f,1.0f,0.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO3 = object3D.at(index);
    index++;

    object3D.push_back( new object(gltfModel.at(5).size(),gltfModel.at(5).data()) );
    graphics->bindBaseObject(object3D.at(index));
    object3D.at(index)->setColor(glm::vec4(0.0f,1.0f,1.0f,1.0f));
    object3D.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    object *UFO4 = object3D.at(index);
    index++;

    groups.at(0)->translate(glm::vec3(0.0f,0.0f,15.0f));
    groups.at(0)->addObject(Box0);

    groups.at(2)->translate(glm::vec3(5.0f,0.0f,5.0f));
    groups.at(2)->addObject(UFO1);

    groups.at(3)->translate(glm::vec3(-5.0f,0.0f,5.0f));
    groups.at(3)->addObject(UFO2);

    groups.at(4)->translate(glm::vec3(10.0f,0.0f,5.0f));
    groups.at(4)->addObject(UFO3);

    groups.at(5)->translate(glm::vec3(-10.0f,0.0f,5.0f));
    groups.at(5)->addObject(UFO4);
}

void scene2::mouseEvent(GLFWwindow* window, float frameTime)
{
    static_cast<void>(frameTime);

    double x, y;

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        if(!cameraAnimation)
        {
            double sensitivity = 0.001;
            glfwGetCursorPos(window,&x,&y);
            angx = sensitivity*(x - xMpos);
            angy = sensitivity*(y - yMpos);
            xMpos = x;
            yMpos = y;
            cameras->rotateX(angy,glm::vec3(1.0f,0.0f,0.0f));
            cameras->rotateY(angx,glm::vec3(0.0f,0.0f,1.0f));
            graphics->resetUboWorld();
        }
    }
    else
    {
        glfwGetCursorPos(window,&xMpos,&yMpos);
    }


}

void scene2::keyboardEvent(GLFWwindow* window, float frameTime)
{
    if(!cameraAnimation)
    {
        float sensitivity = 5.0f*frameTime;
        if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
        {
            float x = -sensitivity*cameras->getViewMatrix()[0][2];
            float y = -sensitivity*cameras->getViewMatrix()[1][2];
            float z = -sensitivity*cameras->getViewMatrix()[2][2];
            cameras->translate(glm::vec3(x,y,z));
            graphics->resetUboWorld();
        }
        if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
        {
            float x = sensitivity*cameras->getViewMatrix()[0][2];
            float y = sensitivity*cameras->getViewMatrix()[1][2];
            float z = sensitivity*cameras->getViewMatrix()[2][2];
            cameras->translate(glm::vec3(x,y,z));
            graphics->resetUboWorld();
        }
        if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
        {
            float x = -sensitivity*cameras->getViewMatrix()[0][0];
            float y = -sensitivity*cameras->getViewMatrix()[1][0];
            float z = -sensitivity*cameras->getViewMatrix()[2][0];
            cameras->translate(glm::vec3(x,y,z));
            graphics->resetUboWorld();
        }
        if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
        {
            float x = sensitivity*cameras->getViewMatrix()[0][0];
            float y = sensitivity*cameras->getViewMatrix()[1][0];
            float z = sensitivity*cameras->getViewMatrix()[2][0];
            cameras->translate(glm::vec3(x,y,z));
            graphics->resetUboWorld();
        }
        if(glfwGetKey(window,GLFW_KEY_Z) == GLFW_PRESS)
        {
            float x = sensitivity*cameras->getViewMatrix()[0][1];
            float y = sensitivity*cameras->getViewMatrix()[1][1];
            float z = sensitivity*cameras->getViewMatrix()[2][1];
            cameras->translate(glm::vec3(x,y,z));
            graphics->resetUboWorld();
        }
        if(glfwGetKey(window,GLFW_KEY_X) == GLFW_PRESS)
        {
            float x = -sensitivity*cameras->getViewMatrix()[0][1];
            float y = -sensitivity*cameras->getViewMatrix()[1][1];
            float z = -sensitivity*cameras->getViewMatrix()[2][1];
            cameras->translate(glm::vec3(x,y,z));
            graphics->resetUboWorld();
        }

        for(uint32_t i=0;i<2;i++){
            if(glfwGetKey(window,GLFW_KEY_LEFT_CONTROL) == 1 && backStage[i] == GLFW_PRESS && glfwGetKey(window,49+i) == 0){
                dQuat[i] = cameras->getDualQuaternion();
                quatX[i] = cameras->getquatX();
                quatY[i] = cameras->getquatY();
                std::cout<<dQuat[i]<<std::endl;
            }
        }

        for(uint32_t i=0;i<2;i++){
            if( glfwGetKey(window,GLFW_KEY_LEFT_CONTROL) == 0 && backStage[i] == GLFW_PRESS && glfwGetKey(window,49+i) == 0){
                cameras->setDualQuaternion(dQuat[i]);
                cameras->setQuaternions(quatX[i],quatY[i]);
                cameraPoint = i+1;
                graphics->resetUboWorld();
                std::cout<<dQuat[i]<<std::endl;
            }
        }

        for(uint32_t i=0;i<2;i++){
            backStage[i] = glfwGetKey(window,49+i);
        }
    }

    if(backStageSpace == GLFW_PRESS && glfwGetKey(window,GLFW_KEY_SPACE) == 0){
        cameraAnimation = true;
    }
    backStageSpace = glfwGetKey(window,GLFW_KEY_SPACE);

    if(glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window,GLFW_TRUE);
    }

}

void scene2::updates(float frameTime)
{
    globalTime += frameTime;

    if(cameraAnimation){
        cameraTimer += frameTime;
        float t = cameraTimer/cameraAnimationTime;

        for(uint32_t i=0;i<1;i++){
            if(cameraPoint == i+1){
                dualQuaternion<float> Quat = slerp(dQuat[i],dQuat[i+1],t);
                cameras->setDualQuaternion(Quat);
                //cameras->setQuaternion(slerp(quatX[i]*quatY[i],quatX[i+1]*quatY[i+1],t));
            }
        }
        if(cameraPoint == 2){
            dualQuaternion<float> Quat = slerp(dQuat[1],dQuat[0],t);
            cameras->setDualQuaternion(Quat);
            //cameras->setQuaternion(slerp(quatX[1]*quatY[1],quatX[0]*quatY[0],t));
        }

        graphics->resetUboWorld();

        if(cameraTimer>cameraAnimationTime){
            for(uint32_t i=0;i<1;i++){
                if(cameraPoint == i+1){
                    cameras->setDualQuaternion(dQuat[i+1]);
                    cameras->setQuaternions(quatX[i+1],quatY[i+1]);
                    cameraPoint++;
                    break;
                }else  if(cameraPoint == 2){
                    cameras->setDualQuaternion(dQuat[0]);
                    cameras->setQuaternions(quatX[0],quatY[0]);
                    cameraPoint = 1;
                }
            }
            cameraTimer = 0.0f;
            cameraAnimation = false;
        }
    }
}

