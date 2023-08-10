#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <sstream>
#include <utility>
#include <filesystem>


#ifdef WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define STB_IMAGE_IMPLEMENTATION

#include <stb_image.h>
#include <vulkan.h>
#include <glfw3.h>
// #include <imgui.h>
// #include <imgui_impl_glfw.h>
// #include <imgui_impl_vulkan.h>

#include <graphicsManager.h>
#include <vector.h>
#include <matrix.h>

#include "testScene.h"
// #include "testPos.h"

bool framebufferResized = false;

// // Data
// static VkAllocationCallbacks*   g_Allocator = nullptr;
// static VkPipelineCache          g_PipelineCache = VK_NULL_HANDLE;
// static VkDescriptorPool         g_DescriptorPool = VK_NULL_HANDLE;
//
// static ImGui_ImplVulkanH_Window g_MainWindowData;
// static int                      g_MinImageCount = 3;
//
// static void check_vk_result(VkResult err)
// {
//     if (err == 0)
//         return;
//     fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
//     if (err < 0)
//         abort();
// }

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName = "");
std::pair<uint32_t,uint32_t> resize(GLFWwindow* window, graphicsManager* app, scene* testScene);

int main()
{
    float fps = 60.0f;
    bool fpsLock = false;
    uint32_t WIDTH = 800;
    uint32_t HEIGHT = 800;
    const std::filesystem::path ExternalPath = std::filesystem::absolute(std::string(__FILE__) + "/../../");

    GLFWwindow* window = initializeWindow(WIDTH, HEIGHT, ExternalPath / "dependences/texture/icon.png");

    graphicsManager app;
    debug::checkResult(app.createSurface(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createDevice(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createSwapChain(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createLinker(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createSyncObjects(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));

        // {
        //     VkDescriptorPoolSize pool_sizes[] =
        //     {
        //         { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        //     };
        //     VkDescriptorPoolCreateInfo pool_info = {};
        //     pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        //     pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        //     pool_info.maxSets = 1;
        //     pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        //     pool_info.pPoolSizes = pool_sizes;
        //     check_vk_result(vkCreateDescriptorPool(app.getDevice().getLogical(), &pool_info, g_Allocator, &g_DescriptorPool));
        // }

        // // Setup Dear ImGui context
        // IMGUI_CHECKVERSION();
        // ImGui::CreateContext();
        // ImGuiIO& io = ImGui::GetIO(); (void)io;
        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

        // // Setup Dear ImGui style
        // ImGui::StyleColorsDark();

        // // Setup Platform/Renderer backends
        // ImGui_ImplGlfw_InitForVulkan(window, true);
        // ImGui_ImplVulkan_InitInfo init_info = {};
        // init_info.Instance = app.getInstance();
        // init_info.PhysicalDevice = app.getDevice().instance;
        // init_info.Device = app.getDevice().getLogical();
        // init_info.QueueFamily = 0;
        // init_info.Queue = app.getDevice().getQueue(0,0);
        // init_info.PipelineCache = g_PipelineCache;
        // init_info.DescriptorPool = g_DescriptorPool;
        // init_info.Subpass = 0;
        // init_info.MinImageCount = g_MinImageCount;
        // init_info.ImageCount = 3;
        // init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        // init_info.Allocator = g_Allocator;
        // init_info.CheckVkResultFn = check_vk_result;
        // //ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

    testScene testScene(&app, window, ExternalPath);
    testScene.create(WIDTH,HEIGHT);

//    bool show_demo_window = true;
//    bool show_another_window = false;
//    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    static auto pastTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window))
    {
        float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::high_resolution_clock::now() - pastTime).count();

        if(fpsLock && fps < 1.0f/frameTime) continue;
        pastTime = std::chrono::high_resolution_clock::now();

//        ImGui_ImplGlfw_NewFrame();
//        ImGui::NewFrame();
//        {
//            static float f = 0.0f;
//            static int counter = 0;
//
//            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
//
//            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
//            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
//            ImGui::Checkbox("Another Window", &show_another_window);
//
//            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
//            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color
//
//            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
//                counter++;
//            ImGui::SameLine();
//            ImGui::Text("counter = %d", counter);
//
//            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
//            ImGui::End();
//        }

//        // Rendering
//        ImGui::Render();
//        ImDrawData* draw_data = ImGui::GetDrawData();
//        const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
//        if (!is_minimized)
//        {
//            wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
//            wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
//            wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
//            wd->ClearValue.color.float32[3] = clear_color.w;
//            //FrameRender(wd, draw_data);
//            //FramePresent(wd);
//        }

        glfwSetWindowTitle(window, std::stringstream("Vulkan [" + std::to_string(1.0f/frameTime) + " FPS]").str().c_str());

        if(app.checkNextFrame() != VK_ERROR_OUT_OF_DATE_KHR)
        {
            testScene.updateFrame(app.getImageIndex(),frameTime,WIDTH,HEIGHT);

            if (VkResult result = app.drawFrame(); result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
                std::tie(WIDTH, HEIGHT) = resize(window,&app,&testScene);
            } else if(result) {
                throw std::runtime_error("failed to with " + std::to_string(result));
            }
        }
    }

    debug::checkResult(app.deviceWaitIdle(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));

    testScene.destroy();
    app.destroySwapChain();
    app.destroyLinker();
    app.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

std::pair<uint32_t,uint32_t> resize(GLFWwindow* window, graphicsManager* app, scene* testScene)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width * height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    app->deviceWaitIdle();
    app->destroySwapChain();
    app->destroyLinker();
    app->createSwapChain(window);
    app->createLinker();

    testScene->resize(width, height);

    framebufferResized = false;

    return std::pair<uint32_t,uint32_t>(width, height);
}

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int){ framebufferResized = true;});

    if(iconName.string().size() > 0){
        int width, height, comp;
        stbi_uc* img = stbi_load(iconName.string().c_str(), &width, &height, &comp, 0);
        GLFWimage images{width,height,img};
        glfwSetWindowIcon(window,1,&images);
    }

    return window;
}
