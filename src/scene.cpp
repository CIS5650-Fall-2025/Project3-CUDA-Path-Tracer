#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "scene.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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
    else if (ext == ".obj")
    {
        std::unordered_map<std::string, uint32_t> MatNameToID;
        if (loadFromObj(filename, false, MatNameToID) == 1)
        {
            cout << "Couldn't read from " << filename << endl;
            exit(-1);
        }
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

int Scene::loadFromObj(const std::string& filePath, bool withinJsonFile, std::unordered_map<std::string, uint32_t>& MatNameToID)
{
    tinyobj::attrib_t _attrib;
    std::vector<tinyobj::shape_t> _shapes;
    std::vector<tinyobj::material_t> _materials;

    std::string warn;
    std::string err;

    // Load the OBJ file
    bool ret = tinyobj::LoadObj(&_attrib, &_shapes, &_materials, &warn, &err, filePath.c_str(), "../scenes");

    if (!err.empty()) {
        // If there's an error, print it
        std::cerr << err << std::endl;
    }

    if (!ret) {
        std::cerr << "Failed to load/parse the .obj file." << std::endl;
        return 1;
    }

    std::cout << "OBJ loader..." << std::endl;
    std::cout << "Loaded " << _materials.size() << " materials." << std::endl;
    std::cout << "Loaded " << _shapes.size() << " shapes." << std::endl;
    std::cout << "Loaded " << _attrib.vertices.size() << " vertices." << std::endl;
    int triangleCount = 0;

    for (const auto& material : _materials)
    {
        Material newMaterial{};

        if (material.emission[0] > 0 || material.emission[1] > 0 || material.emission[2] > 0)
        {
            newMaterial.color = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
            newMaterial.emittance = glm::length(newMaterial.color);
            newMaterial.color = glm::normalize(newMaterial.color);
            if (material.illum > 0)
            {
                newMaterial.emittance *= material.illum;
            }
        }
        else
        {
            newMaterial.color = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
            newMaterial.emittance = 0.0f;
        }

        newMaterial.hasRefractive = (material.transmittance[0] > 0 || material.transmittance[1] > 0 || material.transmittance[2] > 0);
        if (newMaterial.hasRefractive)
        {
            newMaterial.indexOfRefraction = material.ior;
            newMaterial.hasReflective = 1.0f;
        }
        else
        {
            newMaterial.hasReflective = (material.specular[0] > 0 || material.specular[1] > 0 || material.specular[2] > 0);
            if (newMaterial.hasReflective)
            {
                newMaterial.roughness = material.roughness;
            }
        }

        MatNameToID[material.name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    for (const auto& shape : _shapes)
    {
        triangleCount += shape.mesh.indices.size() / 3;
        // Parse positions and normals from the attrib array based on indices
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            Geom newGeom;
            newGeom.type = TRIANGLE;
            newGeom.materialid = MatNameToID[shape.name];

            glm::vec3 v0, v1, v2, n0, n1, n2;

            // Get three consecutive indices that form a triangle
            tinyobj::index_t idx0 = shape.mesh.indices[i];
            tinyobj::index_t idx1 = shape.mesh.indices[i + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[i + 2];

            // Get vertex positions
            v0 = glm::vec3(
                _attrib.vertices[3 * idx0.vertex_index + 0],
                _attrib.vertices[3 * idx0.vertex_index + 1],
                _attrib.vertices[3 * idx0.vertex_index + 2]);
            v1 = glm::vec3(
                _attrib.vertices[3 * idx1.vertex_index + 0],
                _attrib.vertices[3 * idx1.vertex_index + 1],
                _attrib.vertices[3 * idx1.vertex_index + 2]);
            v2 = glm::vec3(
                _attrib.vertices[3 * idx2.vertex_index + 0],
                _attrib.vertices[3 * idx2.vertex_index + 1],
                _attrib.vertices[3 * idx2.vertex_index + 2]);

            // Get normals if available
            if (idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0) {
                n0 = glm::vec3(
                    _attrib.normals[3 * idx0.normal_index + 0],
                    _attrib.normals[3 * idx0.normal_index + 1],
                    _attrib.normals[3 * idx0.normal_index + 2]);
                n1 = glm::vec3(
                    _attrib.normals[3 * idx1.normal_index + 0],
                    _attrib.normals[3 * idx1.normal_index + 1],
                    _attrib.normals[3 * idx1.normal_index + 2]);
                n2 = glm::vec3(
                    _attrib.normals[3 * idx2.normal_index + 0],
                    _attrib.normals[3 * idx2.normal_index + 1],
                    _attrib.normals[3 * idx2.normal_index + 2]);
            }
            else {
                // Compute normals from the geometry of the triangle
                glm::vec3 edge1 = v1 - v0;
                glm::vec3 edge2 = v2 - v0;
                glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

                // Add the same normal to all vertices of the triangle
                n0 = normal;
                n1 = normal;
                n2 = normal;
            }

            // Store the vertices and normals in the Geom struct
            newGeom.v0 = v0;
            newGeom.v1 = v1;
            newGeom.v2 = v2;
            newGeom.n0 = n0;
            newGeom.n1 = n1;
            newGeom.n2 = n2;

            newGeom.translation = glm::vec3(0.0f);
            newGeom.rotation = glm::vec3(0.0f);
            newGeom.scale = glm::vec3(1.0f);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }
    std::cout << "Loaded " << triangleCount << " triangles." << std::endl;

    if (withinJsonFile == false)
    {
        Camera& camera = state.camera;
        RenderState& state = this->state;
        camera.resolution.x = 800;
        camera.resolution.y = 800;
        float fovy = 45.0f;
        state.iterations = 5000;
        state.traceDepth = 8;
        state.imageName = filePath;
        camera.position = glm::vec3(0.0, 5.0, 10.5);
        camera.lookAt = glm::vec3(0.0, 5.0, 0.0);
        camera.up = glm::vec3(0.0, 1.0, 0.0);

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

        if (!geoms.empty())
        {
            BVHBuilder bvhBuilder = BVHBuilder(geoms, bvhNodes);
            bvhRootIndex = bvhBuilder.buildBVH(geomIdx);
        }
    }

    return 0;
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
        Material newMaterial{};                         // all struct properties are initialized to 0.0f
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
            newMaterial.roughness = glm::clamp((float)p["ROUGHNESS"], 0.0f, 1.0f);
            newMaterial.hasReflective = p["REFLECTIVE"];
            newMaterial.hasRefractive = p["REFRACTIVE"];
            newMaterial.indexOfRefraction = p["INDEXOFREFRACTION"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "obj")
        {
            std::string filename = "../scenes/" + (string)(p["FILENAME"]);
            if (loadFromObj(filename, true, MatNameToID) == 1)
            {
                cout << "Couldn't read from " << filename << endl;
                exit(-1);
            }
        }
        else {
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

    if (!geoms.empty())
    {
        BVHBuilder bvhBuilder = BVHBuilder(geoms, bvhNodes);
        bvhRootIndex = bvhBuilder.buildBVH(geomIdx);
    }
}
