#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename) : useDirectLighting(false)
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

void Scene::loadFromJSON(const std::string &jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto &materialsData = data["Materials"];
    std::vector<std::pair<std::string, Material>> materialNamePairs;
    for (const auto &item : materialsData.items())
    {
        const auto &name = item.key();
        const auto &p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto &col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto &col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto &col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = true;
            newMaterial.specular.color = newMaterial.color;
        }
        else if (p["TYPE"] == "Transmissive")
        {
            const auto &col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = true;
            newMaterial.indexOfRefraction = p.contains("IOR") ? p["IOR"].get<float>() : 1.55f;
        }
        else if (p["TYPE"] == "Fresnel")
        {
            const auto &col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasReflective = true;
            newMaterial.hasRefractive = true;
            newMaterial.indexOfRefraction = p.contains("IOR") ? p["IOR"].get<float>() : 1.55f;
        }
        materialNamePairs.emplace_back(std::make_pair(name, newMaterial));
    }

    // I put the materials and objects that are lights first in the list so we can use the front of the object list as the light list
    std::sort(materialNamePairs.begin(), materialNamePairs.end(),
              [](const std::pair<std::string, Material> &a, const std::pair<std::string, Material> &b)
              { return a.second.emittance > b.second.emittance; });

    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (size_t i = 0; i < materialNamePairs.size(); i++)
    {
        auto [name, material] = materialNamePairs[i];
        materials.push_back(material);
        MatNameToID[name] = i;
    }

    size_t nonEmittingMaterialIndex = std::find_if(materials.cbegin(), materials.cend(),
                                                   [](const Material &material)
                                                   { return material.emittance == 0; }) -
                                      materials.cbegin();

    const auto &objectsData = data["Objects"];
    for (const auto &p : objectsData)
    {
        const auto &type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "square")
        {
            newGeom.type = SQUARE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else
        {
            throw std::invalid_argument("Bad object type");
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto &trans = p["TRANS"];
        const auto &rotat = p["ROTAT"];
        const auto &scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    for (const auto &p : data["Triangles"]) {
        Tri tri;
        tri.materialid = MatNameToID[p["MATERIAL"]];
        const auto &points = p["POINTS"];
        for (size_t i = 0; i < 3; i++) {
            const auto &point = points[i];
            tri.points[i] = glm::vec3(point[0], point[1], point[2]);
        }
        tris.push_back(tri);
    }

    std::sort(geoms.begin(), geoms.end(), [](const Geom &g1, const Geom &g2)
              { return g1.materialid < g2.materialid; });

    numLights = std::find_if(geoms.cbegin(), geoms.cend(),
                             [&](const Geom &geom)
                             { return materials[geom.materialid].emittance == 0; }) -
                geoms.cbegin();

    const auto &cameraData = data["Camera"];
    Camera &camera = state.camera;
    RenderState &state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto &pos = cameraData["EYE"];
    const auto &lookat = cameraData["LOOKAT"];
    const auto &up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);
    camera.lensSize = cameraData.contains("LENSSIZE") ? cameraData["LENSSIZE"].get<float>() : 0.f;
    camera.focalDist = cameraData.contains("FOCALDIST") ? cameraData["FOCALDIST"].get<float>() : 0.f;

    // calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
