#include "main.h"
#include "preview.h"
#include <cstring>
#include <OpenImageDenoise/oidn.hpp>
#include "mathUtils.h"

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

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

OIDNDevice oidnDevice;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
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

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    scene->loadSceneModels();
    scene->buildDevSceneData();

    oidnDevice = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice(oidnDevice);

    // GLFW main loop
    mainLoop();

    oidnReleaseDevice(oidnDevice);

    return 0;
}

void saveImage()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            renderState->image[index] = pix / samples;
        }
    }
    
    OIDNBuffer colorBuf = oidnNewBuffer(oidnDevice, width * height * 3 * sizeof(float));
    OIDNBuffer albedoBuf = oidnNewBuffer(oidnDevice, width * height * 3 * sizeof(float));
    OIDNBuffer normalBuf = oidnNewBuffer(oidnDevice, width * height * 3 * sizeof(float));
    OIDNFilter filter = oidnNewFilter(oidnDevice, "RT");

    oidnSetFilterImage(filter, "color", colorBuf,
        OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterImage(filter, "albedo", albedoBuf,
        OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterImage(filter, "normal", normalBuf,
        OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterImage(filter, "output", colorBuf,
        OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterBool(filter, "hdr", true);
    oidnCommitFilter(filter);

    float* colorArr = (float*)oidnGetBufferData(colorBuf);
    float* albedoArr = (float*)oidnGetBufferData(albedoBuf);
    float* normalArr = (float*)oidnGetBufferData(normalBuf);

    size_t imgSize = width * height * 3 * sizeof(float);
    memcpy(colorArr, renderState->image.data(), imgSize);
    memcpy(albedoArr, renderState->albedo.data(), imgSize);
    memcpy(normalArr, renderState->normal.data(), imgSize);

    oidnExecuteFilter(filter);

    const char* errorMessage;
    if (oidnGetDeviceError(oidnDevice, &errorMessage) != OIDN_ERROR_NONE)
        printf("Error: %s\n", errorMessage);

    memcpy(renderState->image.data(), colorArr, imgSize);
    oidnReleaseBuffer(colorBuf);
    oidnReleaseBuffer(albedoBuf);
    oidnReleaseBuffer(normalBuf);
    oidnReleaseFilter(filter);
    
    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            pix = math::ACESMapping(pix);
            pix = math::gammaCorrect(pix);
            img.setPixel(width - 1 - x, y, glm::vec3(pix));
        }
    }

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
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

        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
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
	        case GLFW_KEY_P:
	            saveImage();
	            break;
	        case GLFW_KEY_SPACE:
            {
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
            }
            case GLFW_KEY_Z:
            {
                camchanged = true;
                zoom += 3.f;
                break;
            }
            case GLFW_KEY_X:
            {
                camchanged = true;
                zoom -= 3.f;
                zoom = std::fmax(0.1f, zoom);
                break;
            }
            case GLFW_KEY_W:
            {
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                glm::vec3 forward = cam.view;
                forward.y = 0.0f;
                forward = glm::normalize(forward);
                glm::vec3 right = cam.right;
                right.y = 0.0f;
                right = glm::normalize(right);
                glm::vec3 up = glm::normalize(glm::cross(forward, right));
                cam.lookAt -= up * 2.f;
                camchanged = true;
                break;
            }
            case GLFW_KEY_S:
            {
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                glm::vec3 forward = cam.view;
                forward.y = 0.0f;
                forward = glm::normalize(forward);
                glm::vec3 right = cam.right;
                right.y = 0.0f;
                right = glm::normalize(right);
                glm::vec3 up = glm::normalize(glm::cross(forward, right));
                cam.lookAt += up * 2.f;
                camchanged = true;
                break;
            }
            case GLFW_KEY_MINUS:
            {
                scene->state.camera.aperture = glm::max(0.05f,
                    scene->state.camera.aperture - 0.05f);
                camchanged = true;
                break;
            }
            case GLFW_KEY_EQUAL:
            {
                scene->state.camera.aperture += 0.05f;
                camchanged = true;
                break;
            }
            case GLFW_KEY_F:
            {
                scene->mouseClickPos = glm::vec2(width - lastX, lastY);
                camchanged = true;
                break;
            }
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
        zoom += 10.f * (ypos - lastY) / height;
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

        cam.lookAt -= (float)(xpos - lastX) * right * 0.04f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.04f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
