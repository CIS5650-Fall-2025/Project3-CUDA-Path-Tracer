#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}





glm::vec2 calculatePixelLength(float xscale, float yscale, glm::ivec2 resolution) {
    float pixelWidth = (2 * xscale) / static_cast<float>(resolution.x);
    float pixelHeight = (2 * yscale) / static_cast<float>(resolution.y);
    return glm::vec2(pixelWidth, pixelHeight);
}


//
// "RES": An array representing the resolution of the output image in pixels.
// "FOVY" : The vertical field of view, in degrees.
// "ITERATIONS" : The number of iterations to refine the image during rendering.
// "DEPTH" : The maximum path tracing depth.
// "FILE" : The filename for the rendered output.
// "EYE" : The position of the camera in world coordinates.
// "LOOKAT" : The point in space the camera is directed at.
// "UP" : The up vector defining the camera's orientation.
void Scene::loadCamera() {
    cout << "Start Loading Camera ..." << endl;
    RenderState& curr_state = this->state;
    Camera& camera = curr_state.camera;
    float fov_y;


    for (int i = 0; i < 5; i++) {
        string curr_line;
        utilityCore::safeGetline(fp_in, curr_line);
        vector<string> tokens = utilityCore::tokenizeString(curr_line);
        if (tokens[0] == "RES") {
            camera.resolution.x = std::stoi(tokens[1]);
            camera.resolution.y = std::stoi(tokens[2]);
        }
        else if (tokens[0] == "FOVY") {
            fov_y = std::stof(tokens[1]);
        }
        else if (tokens[0] == "ITERATIONS") {
            state.iterations = std::stoi(tokens[1]);
        }
        else if (tokens[0] == "DEPTH") {
            state.traceDepth = std::stoi(tokens[1]);
        }
        else if (tokens[0] == "FILE") {
            state.imageName = tokens[1];
        }
        else if (tokens[0] == "EYE") {
            camera.position = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "LOOKAT") {
            camera.lookAt = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "UP") {
            camera.up = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
    }


    //calculate fov_x
    float yscaled = tan(fov_y * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;

    float fov_x = (atan(xscaled) * 180) / PI;

    camera.fov = glm::vec2(fov_x, fov_y);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = calculatePixelLength(xscaled, yscaled, camera.resolution);
    camera.view = glm::normalize(camera.lookAt - camera.position);

    int num_pixels = camera.resolution.x * camera.resolution.y;
    curr_state.image.resize(num_pixels);

    std::fill(curr_state.image.begin(), curr_state.image.end(), glm::vec3());

    cout << "Camera Loaded" << endl;
}


