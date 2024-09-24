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
    std::cout << "Reading scene from " << filename << " ..." << endl;
    std::cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        std::cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    //Load Material
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.diffuse = 1.f;
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
            newMaterial.specularRoughness = p["ROUGHNESS"];
        }
        else if (p["TYPE"] == "Refract") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
        }

        if (p.contains("PROCEDUALID")) {
            newMaterial.procedualTextureID = p["PROCEDUALID"];
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    //Load objects
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
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

        const auto& type = p["TYPE"];
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if(type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "obj") 
        {
            newGeom.type = MESH;
            auto obj_filename = p["FILE"];
            loadObj(newGeom, obj_filename, jsonName);
        }
        else if (type == "gltf")
        {
            newGeom.type = MESH;
            auto obj_filename = p["FILE"];
            loadGltf(newGeom, obj_filename, jsonName);
        }
        else 
        {
            continue;
        }
        
        geoms.push_back(newGeom);
    }

    //Load camera
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

    if (cameraData.contains("LENSRADIUS") && cameraData.contains("FOCALDIS")) {
        camera.lensRadius = cameraData["LENSRADIUS"];
        camera.focalDistance = cameraData["FOCALDIS"];
    }

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

int Scene::loadGltf(Geom& newGeom, string filename, string scene_filename) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    string scene_directory = scene_filename.substr(0, scene_filename.find_last_of("/\\") + 1);
    string obj_dirname = scene_directory + filename.substr(0, filename.find_last_of("/\\") + 1);
    string gltf_filename = scene_directory + filename;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltf_filename.c_str());
    if (!warn.empty()) {
        std::cout << "GLTF load warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "GLTF load error: " << err << std::endl;
        return -1;
    }
    if (!ret) {
        std::cerr << "Failed to load GLTF file." << std::endl;
        return -1;
    }

    std::cout << "Loaded GLTF file successfully." << std::endl;

    // Process each mesh in the GLTF file
    for (const auto& mesh : model.meshes) {
        std::cout << "Processing mesh: " << mesh.name << std::endl;
        for (const auto& primitive : mesh.primitives) {
            //Put position data into vector
            int posAccessorIndex = primitive.attributes.at("POSITION");
            tinygltf::Accessor& posAccessor = model.accessors[posAccessorIndex];
            tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            tinygltf::Buffer& positionBuffer = model.buffers[posBufferView.buffer];
            const float* positions = reinterpret_cast<const float*>(&(positionBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]));
            int vStartIdx = vertices.size();
            for (size_t i = 0; i < posAccessor.count; i++) {
                glm::vec3 position(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);
                position = glm::vec3(newGeom.transform * glm::vec4(position, 1.0f));
                vertices.push_back(position);
            }
            cout << "Read " << vertices.size() << " vertices." << endl;

            //Put normal data into vector
            int norAccessorIndex = primitive.attributes.at("NORMAL");
            tinygltf::Accessor& norAccessor = model.accessors[norAccessorIndex];
            tinygltf::BufferView& norBufferView = model.bufferViews[norAccessor.bufferView];
            tinygltf::Buffer& normalBuffer = model.buffers[norBufferView.buffer];
            const float* normalsPtr = reinterpret_cast<const float*>(&(normalBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset]));
            int vnStartIdx = normals.size();
            for (size_t i = 0; i < norAccessor.count; i++) {
                glm::vec3 normal(normalsPtr[i * 3 + 0], normalsPtr[i * 3 + 1], normalsPtr[i * 3 + 2]);
                normal = glm::normalize(glm::vec3(newGeom.invTranspose * glm::vec4(normal, 0.0f)));
                normals.push_back(normal);
            }
            cout << "Read " << normals.size() << " normals." << endl;

            //Put texcoord data into vector
            int vtStartIdx = texcoords.size();
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                int albedoAccessorIndex = primitive.attributes.at("TEXCOORD_0");
                tinygltf::Accessor& albedoAccessor = model.accessors[albedoAccessorIndex];
                tinygltf::BufferView& albedoBufferView = model.bufferViews[albedoAccessor.bufferView];
                tinygltf::Buffer& albedoBuffer = model.buffers[albedoBufferView.buffer];
                const float* uv = reinterpret_cast<const float*>(&(albedoBuffer.data[albedoBufferView.byteOffset + albedoAccessor.byteOffset]));
                for (size_t i = 0; i < albedoAccessor.count; ++i) {
                    glm::vec2 texcoord(uv[i * 2 + 0], uv[i * 2 + 1]);
                    texcoords.push_back(texcoord);
                }
                cout << "Read " << texcoords.size() - vtStartIdx << " texcoords for primitive." << endl;
            }
            else {
                cout << "TEXCOORD_0 attribute not found for primitive, skipping texcoords." << endl;
            }

            //Generate material
            if (primitive.material >= 0) {
                int index = model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index;
                if (index != -1) {
                    tinygltf::Texture& texture = model.textures[index];
                    string albedoMapPath = "../assets/" + model.images[texture.source].uri;
                    cout << "Texture path: " << albedoMapPath << endl;
                }
                else {
                    cout << "No texture material for this mesh." << endl;
                }
            }
            else {
                cout << "No texture material for this mesh." << endl;
            }

            //Generate triangles
            if (primitive.indices >= 0) {
                tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                auto type = indexAccessor.componentType;
                if (indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT && indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    cout << "Unsupported index type for the give glTF model." << endl;
                    return -1;
                }
                newGeom.meshidx = meshes.size();
                uint32_t* indices = reinterpret_cast<uint32_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                for (size_t i = 0; i < indexAccessor.count; i += 3) {
                    Mesh newMesh;

                    newMesh.v[0] = indices[i] + vStartIdx;
                    newMesh.v[1] = indices[i + 1] + vStartIdx;
                    newMesh.v[2] = indices[i + 2] + vStartIdx;

                    if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                        newMesh.vn[0] = indices[i] + vnStartIdx;
                        newMesh.vn[1] = indices[i + 1] + vnStartIdx;
                        newMesh.vn[2] = indices[i + 2] + vnStartIdx;
                    }

                    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                        newMesh.vt[0] = indices[i] + vtStartIdx;
                        newMesh.vt[1] = indices[i + 1] + vtStartIdx;
                        newMesh.vt[2] = indices[i + 2] + vtStartIdx;
                    }
                    else
                    {
                        newMesh.vt[0] = -1;
                        newMesh.vt[1] = -1;
                        newMesh.vt[2] = -1;
                    }
                    newMesh.materialid = newGeom.materialid;

                    newMesh.aabb.min = glm::min(vertices[newMesh.v[0]], glm::min(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
                    newMesh.aabb.max = glm::max(vertices[newMesh.v[0]], glm::max(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
                    newMesh.aabb.centroid = (newMesh.aabb.min + newMesh.aabb.max) * 0.5f;

                    meshes.push_back(newMesh);


                }
                newGeom.meshcnt = meshes.size() - newGeom.meshidx;
                cout << "Number of meshes: " << newGeom.meshcnt << endl;
                newGeom.bvhrootidx = buildBVHEqualCount(newGeom.meshidx, newGeom.meshidx + newGeom.meshcnt);
            }

        }
    }
    return 1;
}

void Scene::loadObj(Geom& newGeom, string obj_filename, string scene_filename)
{
    string scene_directory = scene_filename.substr(0, scene_filename.find_last_of("/\\") + 1);
    string obj_dirname = scene_directory + obj_filename.substr(0, obj_filename.find_last_of("/\\") + 1);
    obj_filename = scene_directory + obj_filename;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tinyobj_materials;

    std::string err;
    bool ret = tinyobj::LoadObj(
        &attrib, &shapes, &tinyobj_materials, &err, obj_filename.c_str(), obj_dirname.c_str(), true);
    if (!err.empty())
    {
        std::cerr << err << std::endl;
    }
    if (!ret)
    {
        std::cerr << "Failed to load/parse .obj." << std::endl;
        exit(1);
    }

    // add materials    
    /*int materialStartIdx = materials.size();
    if (tinyobj_materials.size() > 0)
    {
        for (const tinyobj::material_t& material : tinyobj_materials)
        {
            Material newMaterial;
            if (material.emission[0] + material.emission[1] + material.emission[2] > 0.0f)
            {
                newMaterial.albedo = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
                newMaterial.emittance = 1.0f;
            }
            else
            {
                newMaterial.albedo = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
                newMaterial.emittance = 0.0f;
            }
            newMaterial.metallic = material.metallic;
            newMaterial.roughness = material.roughness;
            newMaterial.ior = material.ior;
            newMaterial.opacity = material.dissolve;
            materials.push_back(newMaterial);
        }
    }*/

    // add vertices
    int vStartIdx = vertices.size();
    for (int i = 0; i < attrib.vertices.size() / 3; i++)
    {
        vertices.push_back(glm::vec3(
            newGeom.transform * glm::vec4(attrib.vertices[3 * i + 0],
                attrib.vertices[3 * i + 1],
                attrib.vertices[3 * i + 2], 1.0f)));
    }

    // add normals
    int vnStartIdx = normals.size();
    for (int i = 0; i < attrib.normals.size() / 3; i++)
    {
        normals.push_back(glm::normalize(glm::vec3(
            newGeom.transform * glm::vec4(attrib.normals[3 * i + 0],
                attrib.normals[3 * i + 1],
                attrib.normals[3 * i + 2], 0.0f))));
    }

    // add texcoords
    int vtStartIdx = texcoords.size();
    for (int i = 0; i < attrib.texcoords.size() / 2; i++)
    {
        texcoords.push_back(glm::vec2(attrib.texcoords[2 * i + 0],
            attrib.texcoords[2 * i + 1]));
    }

    // add meshes
    newGeom.meshidx = meshes.size();
    for (const tinyobj::shape_t& shape : shapes)
    {
        for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++)
        {
            const tinyobj::index_t& idx0 = shape.mesh.indices[3 * f + 0];
            const tinyobj::index_t& idx1 = shape.mesh.indices[3 * f + 1];
            const tinyobj::index_t& idx2 = shape.mesh.indices[3 * f + 2];

            Mesh newMesh;

            newMesh.v[0] = idx0.vertex_index + vStartIdx;
            newMesh.v[1] = idx1.vertex_index + vStartIdx;
            newMesh.v[2] = idx2.vertex_index + vStartIdx;

            newMesh.vn[0] = idx0.normal_index + vnStartIdx;
            newMesh.vn[1] = idx1.normal_index + vnStartIdx;
            newMesh.vn[2] = idx2.normal_index + vnStartIdx;

            newMesh.vt[0] = idx0.texcoord_index + vtStartIdx;
            newMesh.vt[1] = idx1.texcoord_index + vtStartIdx;
            newMesh.vt[2] = idx2.texcoord_index + vtStartIdx;

            /*newMesh.materialid = shape.mesh.material_ids[f] < 0
                ? newGeom.materialid
                : shape.mesh.material_ids[f] + materialStartIdx;*/

            // compute aabb
            newMesh.aabb.min = glm::min(vertices[newMesh.v[0]], glm::min(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
            newMesh.aabb.max = glm::max(vertices[newMesh.v[0]], glm::max(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
            newMesh.aabb.centroid = (newMesh.aabb.min + newMesh.aabb.max) * 0.5f;

            meshes.push_back(newMesh);
        }
    }
    newGeom.meshcnt = meshes.size() - newGeom.meshidx;

    // build bvh
    std::cout << "Building BVH..." << std::endl;
    newGeom.bvhrootidx = buildBVHEqualCount(newGeom.meshidx, newGeom.meshidx + newGeom.meshcnt);
    std::cout << "Built success, node num: " << bvh.size() << std::endl;

    std::cout << endl;
    std::cout << "Loaded " << obj_filename << endl;
    std::cout << "number of vertices: " << attrib.vertices.size() / 3 << endl;
    std::cout << "number of normals: " << attrib.normals.size() / 3 << endl;
    std::cout << "number of texcoords: " << attrib.texcoords.size() / 2 << endl;
    std::cout << "number of meshes: " << newGeom.meshcnt << endl;
    std::cout << "number of materials: " << tinyobj_materials.size() << endl;
}

// reference https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
int Scene::buildBVHEqualCount(int meshStartIdx, int meshEndIdx)
{
    // no mesh
    if (meshEndIdx == meshStartIdx)
    {
        std::cout << "No mesh." << std::endl;
        return -1;
    }

    BVHNode node;

    // compute bvh aabb on CPU, expensive but only done once
    for (int i = meshStartIdx; i < meshEndIdx; i++)
    {
        node.aabb.min = glm::min(node.aabb.min, meshes[i].aabb.min);
        node.aabb.max = glm::max(node.aabb.max, meshes[i].aabb.max);
    }
    node.aabb.centroid = (node.aabb.min + node.aabb.max) * 0.5f;

    // one mesh, leaf node
    if (meshEndIdx - meshStartIdx == 1)
    {
        node.left = -1;
        node.right = -1;
        node.meshidx = meshStartIdx;
    }
    // multiple meshes, internal node
    else
    {
        // split method EqualCounts, range is [meshStartIdx, meshEndIdx)
        int mid = (meshStartIdx + meshEndIdx) / 2;
        glm::vec3 diff = node.aabb.max - node.aabb.min;
        int dim = (diff.x > diff.y && diff.x > diff.z) ? 0 : (diff.y > diff.z) ? 1 : 2;
        std::nth_element(meshes.begin() + meshStartIdx, meshes.begin() + mid, meshes.begin() + meshEndIdx,
            [dim](const Mesh& a, const Mesh& b) {
                return (a.aabb.centroid[dim] < b.aabb.centroid[dim]);
            }
        );

        node.left = buildBVHEqualCount(meshStartIdx, mid);
        node.right = buildBVHEqualCount(mid, meshEndIdx);
        node.meshidx = -1;
    }

    bvh.push_back(node);
    
    return bvh.size() - 1;
}