//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <filesystem>
#include <vector>
#include <string>

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

bool firstTimeOpenWindow = true;

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(1000, 1000, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");
    
    // Load a larger font
    ImFontConfig font_config;
    font_config.OversampleH = 2;
    font_config.OversampleV = 2;
    font_config.PixelSnapH = true;
    font_config.SizePixels = 20.0f;  // Increase this value to make the font larger
    io->Fonts->AddFontDefault(&font_config);

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    
    // Set dark mode style
    ImGui::StyleColorsDark();

    // Set default window size and position
    ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);

    ImGui::Begin("Path Tracer Panel");                  // Create a window called "Hello, world!" and append into it.
    
    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    
    // mesh options
    static bool centralize = false;
    static bool showFileBrowser = false;
    static float rotation[3] = {0.0f, 0.0f, 0.0f};
    // render options
    if (ImGui::CollapsingHeader("Mesh Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Centralize Mesh", &scene->autoCentralizeObj);
        ImGui::Spacing();
        if (scene->autoCentralizeObj) {
            ImGui::PushItemWidth(200);
            ImGui::SliderFloat("Rot X", &scene->rotationOffset.x, -180.0f, 180.0f);
            ImGui::SameLine(0, 20);
            ImGui::SliderFloat("Trans X", &scene->translationOffset.x, -5.0f, 5.0f);
            ImGui::SliderFloat("Rot Y", &scene->rotationOffset.y, -180.0f, 180.0f);
            ImGui::SameLine(0, 20);
            ImGui::SliderFloat("Trans Y", &scene->translationOffset.y, -5.0f, 5.0f);
            ImGui::SliderFloat("Rot Z", &scene->rotationOffset.z, -180.0f, 180.0f);
            ImGui::SameLine(0, 20);
            ImGui::SliderFloat("Trans Z", &scene->translationOffset.z, -5.0f, 5.0f);
            ImGui::SliderFloat("Scale", &scene->scaleOffset, -5.0f, 5.0f);
            ImGui::NewLine();
        }

        // Add a static variable to store the current selection
        static int accelerationStructure = 1; // 0: None, 1: BVC, 2: BVH

        ImGui::Text("Acceleration Structure:");
        ImGui::RadioButton("None", &accelerationStructure, 0);
        ImGui::RadioButton("Basic Bounding Volume Culling", &accelerationStructure, 1);
        ImGui::RadioButton("Bounding Volume Hierarchy", &accelerationStructure, 2);
 
        // Update the scene properties based on the selection
        scene->useBasicBVC = (accelerationStructure == 1);
        scene->useBVH = (accelerationStructure == 2);
        if (accelerationStructure == 2)
        {
            ImGui::Indent();
            ImGui::Spacing();
            ImGui::PushItemWidth(400);
            ImGui::SliderInt("Bins to Split Per Axis", &scene->binsToSplit, 1, 200);

            // Radio button to choose between leaf size and depth
            ImGui::Text("BVH Constraint:");
            static int constraintType = 0;
            if (!scene->useLeafSizeNotDepth) constraintType = 1;
            if (ImGui::RadioButton("Use Max Leaf Size", &constraintType, 0))
            {
                scene->useLeafSizeNotDepth = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Use Max Depth", &constraintType, 1))
            {
                scene->useLeafSizeNotDepth = false;
            }

            ImGui::PushItemWidth(400);

            if (scene->useLeafSizeNotDepth)
            {
                ImGui::SliderInt("Max Leaf Size", &scene->max_leaf_size, 1, 2000);
            }
            else
            {
                // Assuming you have a max_depth variable in your scene
                // If not, you'll need to add it
                ImGui::SliderInt("Max Depth", &scene->max_depth, 1, 200);
            }


            ImGui::PopItemWidth();
            ImGui::Unindent();
        }

        ImGui::NewLine();
    }

    if (ImGui::CollapsingHeader("Render Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Use Path Tracing", &scene->renderWithPathTracing);

        ImGui::Checkbox("Sort by Material", &scene->sortByMaterial);
        ImGui::NewLine();
    }
    
    
    ImGui::NewLine();
    if (ImGui::Button("        Load OBJ/Json File       "))
    {
        showFileBrowser = true;
    }
    ImGui::Text("Renderer starts right after scene is imported");
    ImGui::Spacing();
    ImGui::Text("Re-open the program to reset.");
    ImGui::NewLine();
    
    ImGui::Text("Runtime info");
    ImGui::Spacing();
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Last delta Time: %.3f ms", ImGui::GetIO().DeltaTime * 1000.0f);
    ImGui::Spacing();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    if (showFileBrowser)
    {
        ImGui::Begin("File Browser", &showFileBrowser);

        static std::filesystem::path currentPath = std::filesystem::current_path();
        if (firstTimeOpenWindow)
        {
            currentPath = std::filesystem::current_path() / ".." / "obj_files";
            firstTimeOpenWindow = false;
        }
        if (!std::filesystem::exists(currentPath))
        {
            // Fallback to current directory if "../obj_files/" doesn't exist
            currentPath = std::filesystem::current_path();
        }
        static std::vector<std::filesystem::directory_entry> entries;

        if (ImGui::Button(".."))
        {
            currentPath = currentPath.parent_path();
        }

        ImGui::SameLine();
        ImGui::Text("Current Path: %s", currentPath.string().c_str());
        if (ImGui::BeginChild("Files", ImVec2(0, 300), true))
        {
            entries.clear();
            for (const auto& entry : std::filesystem::directory_iterator(currentPath))
            {
                entries.push_back(entry);
            }

            for (const auto& entry : entries)
            {
                const auto& path = entry.path();
                auto filename = path.filename().string();
                
                if (entry.is_directory())
                {
                    if (ImGui::Selectable(filename.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick))
                    {
                        if (ImGui::IsMouseDoubleClicked(0))
                        {
                            currentPath /= path.filename();
                        }
                    }
                }
                else if (path.extension() == ".obj")
                {
                    if (ImGui::Selectable(filename.c_str()))
                    {
                        // TODO: Implement OBJ loading logic here
                        std::cout << "Selected file: " << (currentPath / filename).string() << std::endl;
                        scene->LoadFromFile((currentPath / filename).string());
                        showFileBrowser = false;
                    }
                }
                else if (path.extension() == ".json")
                {
                    if (ImGui::Selectable(filename.c_str()))
                    {
                        std::cout << "Selected file: " << (currentPath / filename).string() << std::endl;
                        scene->LoadFromFile((currentPath / filename).string());
                        showFileBrowser = false;
                    }
                }
            }
        }
        ImGui::EndChild();

        ImGui::End();
    }
    


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
