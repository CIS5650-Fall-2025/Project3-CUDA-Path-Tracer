#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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
        if (p["TYPE"] == "Lambertian" || p["TYPE"] == "Diffuse")
        {
            newMaterial.type = LAMBERTIAN;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.type = EMISSIVE;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Metal" || p["TYPE"] == "Specular")
        {
            newMaterial.type = METAL;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const float& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == "Dielectric" || p["TYPE"] == "Glass")
        {
            newMaterial.type = DIELECTRIC;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const float& ior = p["IOR"];
            newMaterial.indexOfRefraction = ior;
        }
        else
        {
            // Default into black lambertian
            newMaterial.type = LAMBERTIAN;
            newMaterial.color = glm::vec3(0.0f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
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

        if (type == "cube")
        {
            newGeom.type = CUBE;
            newGeom.numVertices = 2;
            newGeom.vertices[0] = glm::vec3(-1.f);
            newGeom.vertices[1] = glm::vec3(1.f);
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            newGeom.numVertices = 2;
            newGeom.vertices[0] = glm::vec3(-1.f);
            newGeom.vertices[1] = glm::vec3(1.f);
        }
        else if (type == "mesh")
        {
            // Based on tinyobjloader example
            // https://github.com/tinyobjloader/tinyobjloader/blob/release/loader_example.cc
            const std::string basepath = "";
            const std::string& filepath = p["FILEPATH"];

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string warn;
            std::string err;

            // Triangulate obj mesh by default
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(),
                basepath.c_str(), true);

            if (!warn.empty()) {
                std::cout << "WARN: " << warn << std::endl;
            }

            if (!err.empty()) {
                std::cerr << "ERR: " << err << std::endl;
            }

            if (!ret) {
                printf("Failed to load/parse .obj.\n");
                continue;
            }

            // Assume each obj has only one mesh
            newGeom.type = TRIANGLE;
            newGeom.numVertices = 3;
            for (int f = 0; f < shapes[0].mesh.num_face_vertices.size(); f++) {
                // Each face is triangulated, loop over each vertex on the face
                for (int v = 0; v < 3; v++) {
                    // Construct glm::vec3 per vertex
                    int vIndex = static_cast<int>(shapes[0].mesh.indices[f * 3 + v].vertex_index);
                    newGeom.vertices[v] = glm::vec3(
                        attrib.vertices[vIndex * 3 + 0],
                        attrib.vertices[vIndex * 3 + 1],
                        attrib.vertices[vIndex * 3 + 2]
                    );
                }
                geoms.push_back(newGeom);
            }
            continue;
        }

        geoms.push_back(newGeom);
    }

    // Assume the scene is static, so we only need to construct BVH once
    int leafSize = 8;
    bvh = BVH(std::move(geoms), leafSize);
    printf("BVH constructed.\n");

    // Copy over reordered geometry data
    geoms = bvh.geoms;
    nodes = bvh.nodes;

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
    const auto& focalLength = cameraData.value("FOCALLENGTH", 1.0f);
    const auto& apertureSize = cameraData.value("APERTURESIZE", 0.0f);
    camera.focalLength = focalLength;
    camera.apertureSize = apertureSize;

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
