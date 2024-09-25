#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

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
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

Geom Scene::loadFromGltf(const std::string& gltfName) {
    cout << "load from gltf" << endl;
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn;
    std::string err;
    bool res = false;
    
    int pos = gltfName.find(".gltf");
    if (pos >= 0) {
        res = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
    }

    pos = gltfName.find(".glb");
    if (pos >= 0) {
        res = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);
    }

    if (!warn.empty()) cout << "WARNING: " << warn << endl;
    if (!err.empty()) cout << "Error: " << err << endl;
    if (!res) {
        cout << "Failed to load gltf file. " << endl;
        Geom defaultGeom;
        defaultGeom.type = SPHERE;
        return defaultGeom;
    }

    Geom newGeom;
    newGeom.type = OBJECT;
    newGeom.triangleIndex = 0;

    for (int m = 0; m < model.meshes.size(); m++)
    {
        tinygltf::Mesh& mesh = model.meshes[m];
        for (int p = 0; p < mesh.primitives.size(); p++)
        {
            tinygltf::Primitive& primitive = mesh.primitives[p];
            tinygltf::Accessor& vertAcc = model.accessors[primitive.attributes.at("POSITION")];
            tinygltf::BufferView& vertBufView = model.bufferViews[vertAcc.bufferView];
            tinygltf::Buffer& vertBuf = model.buffers[vertBufView.buffer];
            float* verts = reinterpret_cast<float*>(&vertBuf.data[vertBufView.byteOffset + vertAcc.byteOffset]);

            const float* nors = nullptr;
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
            {
                const tinygltf::Accessor& norAcc = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::BufferView& norBufView = model.bufferViews[norAcc.bufferView];
                const tinygltf::Buffer& norBuf = model.buffers[norBufView.buffer];
                nors = reinterpret_cast<const float*>(&norBuf.data[norBufView.byteOffset + norAcc.byteOffset]);
            }

            const float* uvs = nullptr;
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
            {
                const tinygltf::Accessor& uvAcc = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                const tinygltf::BufferView& uvBufView = model.bufferViews[uvAcc.bufferView];
                const tinygltf::Buffer& uvBuf = model.buffers[uvBufView.buffer];
                uvs = reinterpret_cast<const float*>(&uvBuf.data[uvBufView.byteOffset + uvAcc.byteOffset]);
            }

            if (primitive.indices >= 0)
            {
                const tinygltf::Accessor& idxAcc = model.accessors[primitive.indices];
                const tinygltf::BufferView& idxBufView = model.bufferViews[idxAcc.bufferView];
                const tinygltf::Buffer& idxBuf = model.buffers[idxBufView.buffer];
                const uint16_t* indices = reinterpret_cast<const uint16_t*>(&idxBuf.data[idxBufView.byteOffset + idxAcc.byteOffset]);

                Triangle triangle;

                for (size_t i = 0; i < idxAcc.count; i += 3)
                {
                    int idx0 = indices[i];
                    int idx1 = indices[i + 1];
                    int idx2 = indices[i + 2];

                    triangle.vertices[0] = glm::vec3(verts[idx0 * 3], verts[idx0 * 3 + 1], verts[idx0 * 3 + 2]);
                    triangle.vertices[1] = glm::vec3(verts[idx1 * 3], verts[idx1 * 3 + 1], verts[idx1 * 3 + 2]);
                    triangle.vertices[2] = glm::vec3(verts[idx2 * 3], verts[idx2 * 3 + 1], verts[idx2 * 3 + 2]);

                    if (nors)
                    {
                        triangle.normals[0] = glm::vec3(nors[idx0 * 3], nors[idx0 * 3 + 1], nors[idx0 * 3 + 2]);
                        triangle.normals[1] = glm::vec3(nors[idx1 * 3], nors[idx1 * 3 + 1], nors[idx1 * 3 + 2]);
                        triangle.normals[2] = glm::vec3(nors[idx2 * 3], nors[idx2 * 3 + 1], nors[idx2 * 3 + 2]);
                    }

                    if (uvs)
                    {
                        triangle.uv2[0] = glm::vec2(uvs[idx0 * 2], uvs[idx0 * 2 + 1]);
                        triangle.uv2[1] = glm::vec2(uvs[idx1 * 2], uvs[idx1 * 2 + 1]);
                    }

                    triangles.push_back(triangle);
                }
            }
            else
            {
                Triangle triangle;

                for (size_t i = 0; i < vertAcc.count; i += 3)
                {
                    int idx0 = i;
                    int idx1 = i + 1;
                    int idx2 = i + 2;

                    triangle.vertices[0] = glm::vec3(verts[idx0 * 3], verts[idx0 * 3 + 1], verts[idx0 * 3 + 2]);
                    triangle.vertices[1] = glm::vec3(verts[idx1 * 3], verts[idx1 * 3 + 1], verts[idx1 * 3 + 2]);
                    triangle.vertices[2] = glm::vec3(verts[idx2 * 3], verts[idx2 * 3 + 1], verts[idx2 * 3 + 2]);

                    if (nors)
                    {
                        triangle.normals[0] = glm::vec3(nors[idx0 * 3], nors[idx0 * 3 + 1], nors[idx0 * 3 + 2]);
                        triangle.normals[1] = glm::vec3(nors[idx1 * 3], nors[idx1 * 3 + 1], nors[idx1 * 3 + 2]);
                        triangle.normals[2] = glm::vec3(nors[idx2 * 3], nors[idx2 * 3 + 1], nors[idx2 * 3 + 2]);
                    }

                    if (uvs)
                    {
                        triangle.uv2[0] = glm::vec2(uvs[idx0 * 2], uvs[idx0 * 2 + 1]);
                        triangle.uv2[1] = glm::vec2(uvs[idx1 * 2], uvs[idx1 * 2 + 1]);
                    }

                    triangles.push_back(triangle);
                }
            }
        }
    }

    newGeom.triangleCount = triangles.size();
    return newGeom;
}

Geom Scene::loadFromObj(const std::string& objName) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warning;
    std::string error;
    bool res = tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, objName.c_str());

    if (!warning.empty()) cout << "WARNING: " << warning << endl;
    if (!error.empty()) cout << "Error: " << error << endl;

    if (!res)
    {
        cout << "Failed to load .obj file. " << endl;
        Geom defaultGeom;
        defaultGeom.type = SPHERE;
        return defaultGeom;
    }

    Geom newGeom;
    newGeom.type = OBJECT;
    newGeom.triangleIndex = 0;

    for (size_t i = 0; i < shapes.size(); i++)
    {
        size_t index_offset = 0;

        Triangle triangle;

        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++)
        {
            for (size_t v = 0; v < 3; v++)
            {
                tinyobj::index_t idx_t = shapes[i].mesh.indices[index_offset + v];
                size_t idx_v = (size_t)idx_t.vertex_index;
                tinyobj::real_t vx = attrib.vertices[3 * idx_v + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx_v + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx_v + 2];
                triangle.vertices[v] = glm::vec3(vx, vy, vz);

                if (idx_t.normal_index >= 0)
                {
                    size_t idx_n = (size_t)idx_t.normal_index;
                    tinyobj::real_t nx = attrib.normals[3 * idx_n + 0];
                    tinyobj::real_t ny = attrib.normals[3 * idx_n + 1];
                    tinyobj::real_t nz = attrib.normals[3 * idx_n + 2];
                    triangle.normals[v] = glm::vec3(nx, ny, nz);
                }

                if (idx_t.texcoord_index >= 0)
                {
                    size_t idx_uv = (size_t)idx_t.texcoord_index;
                    tinyobj::real_t uvx = attrib.texcoords[2 * idx_uv + 0];
                    tinyobj::real_t uvy = attrib.texcoords[2 * idx_uv + 1];
                    triangle.uv2[v] = glm::vec2(uvx, uvy);
                }
            }
            index_offset += 3;
            triangles.push_back(triangle);
        }
    }

    newGeom.triangleCount = triangles.size();
    cout << "finished geoms reading" << endl;
    //geoms.push_back(newGeom);
    return newGeom;
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
        if (p["REFLECT"] > 0.0) {
            newMaterial.hasReflective = 1;
        }
        if (p["REFRACT"] > 0.0) {
            newMaterial.hasRefractive = 1;
            newMaterial.indexOfRefraction = 0.2; // p["REFRACT"];
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

        const auto& objfile = p["OBJFILE"];
        if (strcmp(to_string(objfile).c_str(), "null") != 0) {
            string filename = objfile;
            cout << "load object file: " << filename << endl;
            newGeom = loadFromObj(filename);
        }

        const auto& glfile = p["GLFILE"];
        if (strcmp(to_string(glfile).c_str(), "null") != 0) {
            string filename = glfile;
            cout << "load object file: " << filename << endl;
            newGeom = loadFromGltf(filename);
        }

        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
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
