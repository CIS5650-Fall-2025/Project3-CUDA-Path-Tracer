#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_obj_loader.h"

using json = nlohmann::json;

Scene::~Scene()
{
}

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

std::string normalizePath(const std::string& path) {
    std::string normalizedPath = path;
    std::replace(normalizedPath.begin(), normalizedPath.end(), '\\', '/');
    return normalizedPath;
}

std::string constructObjFilePath(const std::string& basePath, const std::string& meshName) {
    std::string normalizedBasePath = normalizePath(basePath);
    std::string path = normalizedBasePath.substr(0, normalizedBasePath.find_last_of('/')) + "/meshes/" + meshName + ".obj";
    return path;
}


void loadVertices(
    const tinyobj::attrib_t& attrib,
    const tinyobj::shape_t& shape,
    std::vector<Vertex>& vertices
) {
    const auto& faceVertexCounts = shape.mesh.num_face_vertices;
    const auto& indices = shape.mesh.indices;
    std::size_t indexOffset = 0;

    for (std::size_t faceIdx = 0; faceIdx < faceVertexCounts.size(); ++faceIdx) {
        if (faceVertexCounts[faceIdx] != 3) {
            std::cerr << "Error: Non-triangulated face encountered." << std::endl;
            continue;
        }

        for (std::size_t v = 0; v < 3; ++v) {
            const tinyobj::index_t& idx = indices[indexOffset + v];
            Vertex vertex;

            // Vertex positions
            vertex.position = glm::vec3(
                attrib.vertices[3 * idx.vertex_index + 0],
                attrib.vertices[3 * idx.vertex_index + 1],
                attrib.vertices[3 * idx.vertex_index + 2]);

            // Normals
            if (idx.normal_index >= 0) {
                vertex.normal = glm::vec3(
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2]);
            }

            // Texture coordinates
            if (idx.texcoord_index >= 0) {
                vertex.uvTextureCoordinates = glm::vec2(
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    attrib.texcoords[2 * idx.texcoord_index + 1]);
            }

            vertices.push_back(vertex);
        }
        indexOffset += 3;
    }
}

void computeTangents(std::vector<Vertex>& vertices) {
    for (std::size_t i = 0; i < vertices.size(); i += 3) {
        const glm::vec3 edge1 = vertices[i + 1].position - vertices[i].position;
        const glm::vec3 edge2 = vertices[i + 2].position - vertices[i].position;
        const glm::vec2 deltaUV1 = vertices[i + 1].uvTextureCoordinates - vertices[i].uvTextureCoordinates;
        const glm::vec2 deltaUV2 = vertices[i + 2].uvTextureCoordinates - vertices[i].uvTextureCoordinates;

        const float det = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;
        if (det == 0.0f) continue;

        const glm::vec3 rawTangent = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) / det;

        for (std::size_t j = 0; j < 3; ++j) {
            Vertex& v = vertices[i + j];
            v.tangent = glm::normalize(rawTangent - v.normal * glm::dot(rawTangent, v.normal));
        }
    }
}

void handleMesh(const std::string& jsonName, const std::string& meshName) {
    std::string objFilePath = constructObjFilePath(jsonName, meshName);

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objFilePath, tinyobj::ObjReaderConfig{})) {
        if (!reader.Error().empty()) {
            std::cerr << "Error: " << reader.Error() << std::endl;
        }
        std::abort();
    }

    const tinyobj::attrib_t& attrib = reader.GetAttrib();
    const tinyobj::shape_t& shape = *reader.GetShapes().begin();
    std::vector<Vertex> vertices;

    loadVertices(attrib, shape, vertices);
    computeTangents(vertices);

    std::cout << "Loaded " << vertices.size() << " vertices from " << objFilePath << std::endl;
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

        if (p["TYPE"] == "Refractive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
        }      
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

            // Update reflective property of the specular material
            // based off of the roughness
            const float& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0 - roughness;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;

        if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else if (type == "cube")
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

        //====================================================================================================
        // HANDLE CUSTOM MESHES HERE
        if (newGeom.type == MESH) 
        {
            handleMesh(jsonName, p["NAME"]);
        }
        //====================================================================================================

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    /*camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];*/
    camera.resolution.x = 1000;
    camera.resolution.y = 1000;
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
