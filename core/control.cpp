#include "vulkanCore.h"
#include "transformational/object.h"
#include "transformational/group.h"
#include "transformational/camera.h"
#include "transformational/light.h"

void VkApplication::mouseEvent()
{
    double x, y;
    float sensitivity = 5.0f*frameTime;
    int button = glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT);
    glfwSetScrollCallback(window,&VkApplication::scrol);
    if(button == GLFW_PRESS)
    {
        double sensitivity = 0.001;
        glfwGetCursorPos(window,&x,&y);
        angx = sensitivity*(x - xMpos);
        angy = sensitivity*(y - yMpos);
        xMpos = x;
        yMpos = y;
        cam->rotateX(angy,glm::vec3(1.0f,0.0f,0.0f));
        cam->rotateY(angx,glm::vec3(0.0f,0.0f,-1.0f));
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    else
    {
        glfwGetCursorPos(window,&xMpos,&yMpos);
    }
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        float x = -sensitivity*cam->getViewMatrix()[0][2];
        float y = -sensitivity*cam->getViewMatrix()[1][2];
        float z = -sensitivity*cam->getViewMatrix()[2][2];
        cam->translate(glm::vec3(x,y,z));
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        float x = sensitivity*cam->getViewMatrix()[0][2];
        float y = sensitivity*cam->getViewMatrix()[1][2];
        float z = sensitivity*cam->getViewMatrix()[2][2];
        cam->translate(glm::vec3(x,y,z));
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        float x = -sensitivity*cam->getViewMatrix()[0][0];
        float y = -sensitivity*cam->getViewMatrix()[1][0];
        float z = -sensitivity*cam->getViewMatrix()[2][0];
        cam->translate(glm::vec3(x,y,z));
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
    {
        float x = sensitivity*cam->getViewMatrix()[0][0];
        float y = sensitivity*cam->getViewMatrix()[1][0];
        float z = sensitivity*cam->getViewMatrix()[2][0];
        cam->translate(glm::vec3(x,y,z));
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_Z) == GLFW_PRESS)
    {
        float x = sensitivity*cam->getViewMatrix()[0][1];
        float y = sensitivity*cam->getViewMatrix()[1][1];
        float z = sensitivity*cam->getViewMatrix()[2][1];
        cam->translate(glm::vec3(x,y,z));
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_X) == GLFW_PRESS)
    {
        float x = -sensitivity*cam->getViewMatrix()[0][1];
        float y = -sensitivity*cam->getViewMatrix()[1][1];
        float z = -sensitivity*cam->getViewMatrix()[2][1];
        cam->translate(glm::vec3(x,y,z));
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_4) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(0.0f,0.0f,1.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_6) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(0.0f,0.0f,1.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_8) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(1.0f,0.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_5) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(1.0f,0.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_7) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(0.5f),glm::vec3(0.0f,1.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_9) == GLFW_PRESS)
    {
        groups.at(controledGroup)->rotate(glm::radians(-0.5f),glm::vec3(0.0f,1.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(-1.0f,0.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(1.0f,0.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,1.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,-1.0f,0.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_ADD) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,0.0f,1.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
    }
    if(glfwGetKey(window,GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
    {
        groups.at(controledGroup)->translate(sensitivity*glm::vec3(0.0f,0.0f,-1.0f));
        updateCmdLight = true;
        updatedLightFrames = 0;
        updateCmdWorld = true;
        updatedWorldFrames = 0;
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
}

void VkApplication::scrol(GLFWwindow *window, double xoffset, double yoffset)
{
    static_cast<void>(window);
    static_cast<void>(xoffset);
    static_cast<void>(yoffset);
}
