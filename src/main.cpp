#include "main.h"
#include "preview.h"
#include <cstring>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera
int stratifiedSamples = 1;
float focalLength = 1.0f;
float apertureSize = 0.0f;

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        printf("Usage: %s SAVEFILE.txt\n", argv[0]);
        return 1;
    }

    const char* fileName = argv[1];

    bool restart = false;

    // If the file ends with "/", assume it's a directory for txt files
    // and read from the save files
    if (fileName[strlen(fileName)-1] == '/') {
        readState(fileName);

        restart = true;
        scene->restart = true;
        camchanged = false;
    }
    // Otherwise, assume it's a scene file and create a new session
    else
    {
        // Load scene file
        scene = new Scene(fileName);

        // Create Instance for ImGUIData
        guiData = new GuiDataContainer();

        iteration = 0;
    }

    // Set up camera stuff from loaded path tracer settings
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    stratifiedSamples = scene->state.sampleWidth;
    focalLength = scene->state.camera.focalLength;
    apertureSize = scene->state.camera.apertureSize;

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData, stratifiedSamples, focalLength, apertureSize);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop(restart);

    return 0;
}

void saveImage()
{
    retrieveRenderBuffer();

    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

/*
 * The structure of our saveFile will be
 * iteration
 * scene
 * renderstate
 * guidata
 */
void saveState()
{
    retrieveRenderBuffer();

    std::ofstream saveFile;
    saveFile.open("../SaveFileIterSceneRenderGui.txt", std::ofstream::out | std::ofstream::trunc);
    
    // iteration
    saveFile.write(reinterpret_cast<const char*>(&iteration), sizeof(int));

    // scene
    int size = scene->materials.size();
    saveFile.write(reinterpret_cast<const char*>(&size), sizeof(int));
    for (int i = 0; i < size; i++) {
        saveFile.write(reinterpret_cast<const char*>(&scene->materials[i]), sizeof(Material));
    }

    size = scene->geoms.size();
    saveFile.write(reinterpret_cast<const char*>(&size), sizeof(int));
    for (int i = 0; i < size; i++) {
        saveFile.write(reinterpret_cast<const char*>(&scene->geoms[i]), sizeof(Geom));
    }

    saveFile.write(reinterpret_cast<const char*>(&scene->bvhRootIdx), sizeof(int));
    
    size = scene->nodes.size();
    saveFile.write(reinterpret_cast<const char*>(&size), sizeof(int));
    for (int i = 0; i < size; i++) {
        saveFile.write(reinterpret_cast<const char*>(&scene->nodes[i]), sizeof(BVH::Node));
    }

    saveFile.write(reinterpret_cast<const char*>(&scene->bgTextureInfo), sizeof(ImageTextureInfo));

    // render state
    saveFile.write(reinterpret_cast<const char*>(&scene->state.camera), sizeof(Camera));
    saveFile.write(reinterpret_cast<const char*>(&scene->state.iterations), sizeof(int));
    saveFile.write(reinterpret_cast<const char*>(&scene->state.sampleWidth), sizeof(int));
    saveFile.write(reinterpret_cast<const char*>(&scene->state.traceDepth), sizeof(int));
    size = std::strlen(scene->state.imageName.c_str()) + 1;
    saveFile.write(reinterpret_cast<const char*>(&size), sizeof(int));
    saveFile.write(reinterpret_cast<const char*>(&scene->state.imageName), size);

    // guidata
    saveFile.write(reinterpret_cast<const char*>(&guiData->TracedDepth), sizeof(int));

    if (!saveFile)
        throw std::runtime_error("File write error");

    saveFile.close();

    printf("SaveFileIterSceneRenderGui.txt saved\n");

    saveFile.open("../SaveFileImage.txt", std::ofstream::out | std::ofstream::trunc);

    size = scene->state.image.size();
    saveFile << size << "\n";
    for (int i = 0; i < size; i++) {
        saveFile << scene->state.image[i][0] << " " << scene->state.image[i][1] << " " << scene->state.image[i][2] << " " << scene->state.image[i][3] << "\n";
    }

    if (!saveFile)
        throw std::runtime_error("File write error");

    saveFile.close();

    printf("SaveFileImage.txt saved\n");

    saveFile.open("../SaveFileTextures.txt", std::ofstream::out | std::ofstream::trunc);

    size = scene->textures.size();
    saveFile << size << "\n";
    for (int i = 0; i < size; i++) {
        saveFile << scene->textures[i][0] << " " << scene->textures[i][1] << " " << scene->textures[i][2] << " " << scene->textures[i][3] << "\n";
    }

    if (!saveFile)
        throw std::runtime_error("File write error");

    saveFile.close();

    printf("SaveFileTextures.txt saved\n");

    printf("Save file successfully generated\n");
}

void readState(std::string filePath)
{
    std::ifstream readFile(filePath + "SaveFileIterSceneRenderGui.txt");

    // iteration
    readFile.read(reinterpret_cast<char*>(&iteration), sizeof(int));

    // scene
    scene = new Scene();

    int size;
    readFile.read(reinterpret_cast<char*>(&size), sizeof(int));
    scene->materials.resize(size);
    for (int i = 0; i < size; i++) {
        readFile.read(reinterpret_cast<char*>(&scene->materials[i]), sizeof(Material));
    }

    readFile.read(reinterpret_cast<char*>(&size), sizeof(int));
    scene->geoms.resize(size);
    for (int i = 0; i < size; i++) {
        readFile.read(reinterpret_cast<char*>(&scene->geoms[i]), sizeof(Geom));
    }

    readFile.read(reinterpret_cast<char*>(&scene->bvhRootIdx), sizeof(int));
    
    readFile.read(reinterpret_cast<char*>(&size), sizeof(int));
    scene->nodes.resize(size);
    for (int i = 0; i < size; i++) {
        readFile.read(reinterpret_cast<char*>(&scene->nodes[i]), sizeof(BVH::Node));
    }

    readFile.read(reinterpret_cast<char*>(&scene->bgTextureInfo), sizeof(ImageTextureInfo));

    // render state
    readFile.read(reinterpret_cast<char*>(&scene->state.camera), sizeof(Camera));
    readFile.read(reinterpret_cast<char*>(&scene->state.iterations), sizeof(int));
    readFile.read(reinterpret_cast<char*>(&scene->state.sampleWidth), sizeof(int));
    readFile.read(reinterpret_cast<char*>(&scene->state.traceDepth), sizeof(int));
    readFile.read(reinterpret_cast<char*>(&size), sizeof(int));
    readFile.read(reinterpret_cast<char*>(&scene->state.imageName), size);

    // guidata
    guiData = new GuiDataContainer();
    readFile.read(reinterpret_cast<char*>(&guiData->TracedDepth), sizeof(int));

    readFile.close();

    printf("SaveFileIterSceneRenderGui.txt read\n");

    std::string line;

    readFile.open(filePath + "SaveFileImage.txt");

    utilityCore::safeGetline(readFile, line);
    vector<string> tokens = utilityCore::tokenizeString(line);
    size = atoi(tokens[0].c_str());

    scene->state.image.resize(size);
    int i = 0;
    while (readFile.good()) {
        utilityCore::safeGetline(readFile, line);
        if (!line.empty()) {
            tokens = utilityCore::tokenizeString(line);
            float r = atof(tokens[0].c_str());
            float g = atof(tokens[1].c_str());
            float b = atof(tokens[2].c_str());
            glm::vec3 color = glm::vec3(r, g, b);
            scene->state.image[i] = color;
            i++;
        }
    }

    readFile.close();

    printf("SaveFileImage.txt read\n");

    readFile.open(filePath + "SaveFileTextures.txt");

    utilityCore::safeGetline(readFile, line);
    tokens = utilityCore::tokenizeString(line);
    size = atoi(tokens[0].c_str());

    scene->textures.resize(size);
    i = 0;
    while (readFile.good()) {
		utilityCore::safeGetline(readFile, line);
		if (!line.empty()) {
			tokens = utilityCore::tokenizeString(line);
            float r = atof(tokens[0].c_str());
            float g = atof(tokens[1].c_str());
            float b = atof(tokens[2].c_str());
            float a = atof(tokens[3].c_str());
            glm::vec4 color = glm::vec4(r, g, b, a);
            scene->textures[i] = color;
            i++;
		}
    }

    readFile.close();

    printf("SaveFileTextures.txt read\n");

    printf("Read file successfully read\n");
}

void runCuda(bool restart)
{
    if (camchanged)
    {
        iteration = 0;
        renderState->sampleWidth = stratifiedSamples;

        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        cam.focalLength = focalLength;
        cam.apertureSize = apertureSize;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0 || restart)
    {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
	        case GLFW_KEY_ESCAPE:
	            saveImage();
	            glfwSetWindowShouldClose(window, GL_TRUE);
	            break;
	        case GLFW_KEY_S:
	            saveImage();
	            break;
	        case GLFW_KEY_P:
	            saveState();
	            break;
	        case GLFW_KEY_SPACE:
	            camchanged = true;
	            renderState = &scene->state;
	            Camera& cam = renderState->camera;
	            cam.lookAt = ogLookAt;
	            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
    	return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}

void resetRender()
{
    glClear(GL_COLOR_BUFFER_BIT);
    iteration = 0;
    resetRenderBuffer();
}

void getCamera(float& focalLength_, float& apertureSize_)
{
    focalLength_ = focalLength;
    apertureSize_ = apertureSize;
}

void updateSettings(int stratifiedSamples_, float focalLength_, float apertureSize_)
{
    stratifiedSamples = stratifiedSamples_;
    focalLength = focalLength_;
    apertureSize = apertureSize_;
    camchanged = true;
}