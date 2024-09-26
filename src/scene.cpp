#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_obj_loader.h"
#include "bvh.h"
using json = nlohmann::json;

Scene::Scene(std::string filename)
{
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        std::cout << "Couldn't read from " << filename << std::endl;
        exit(-1);
    }
}

Scene::~Scene()
{
    if (sceneDev)
    {
        sceneDev->freeCudaMemory();
        delete sceneDev;
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Lambertian;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.type = Light;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Specular;
        }
        else if (p["TYPE"] == "Microfacet")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Microfacet;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.ior = p["IOR"];
            newMaterial.metallic = p["METALLIC"];
        }
        else if (p["TYPE"] == "MetallicWorkflow")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MetallicWorkflow;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.metallic = p["METALLIC"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        objects.push_back(Object());
        Object& newObject = objects.back();
        if (type == "cube")
        {
            newObject.type = CUBE;
        }
        else if (type == "sphere")
        {
            newObject.type = SPHERE;
        }
        else if (type == "obj")
        {
            newObject.type = TRIANGLE;
            loadObj(newObject, p["PATH"]);
        }
        newObject.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newObject.transforms.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newObject.transforms.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newObject.transforms.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newObject.transforms.transform = utilityCore::buildTransformationMatrix(
            newObject.transforms.translation, newObject.transforms.rotation, newObject.transforms.scale);
        newObject.transforms.inverseTransform = glm::inverse(newObject.transforms.transform);
        newObject.transforms.invTranspose = glm::inverseTranspose(newObject.transforms.transform);
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

bool Scene::loadObj(Object& newObj, const std::string& objPath)
{
    std::cout << "Start loading Obj file: " << objPath << " ..." << std::endl;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string err;
    std::string mtlPath = objPath.substr(0, objPath.find_last_of('/') + 1);
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &mats, &err, objPath.c_str(), mtlPath.c_str());

    if (!err.empty()) std::cerr << err << std::endl;
    if (!ret)  return false;

    bool hasUV = !attrib.texcoords.empty();

    for (const auto& shape : shapes) {
        for (auto idx : shape.mesh.indices) {
            newObj.meshData.vertices.push_back(*((glm::vec3*)attrib.vertices.data() + idx.vertex_index));
            newObj.meshData.normals.push_back(*((glm::vec3*)attrib.normals.data() + idx.normal_index));
            newObj.meshData.uvs.push_back(hasUV ?
                *((glm::vec2*)attrib.texcoords.data() + idx.texcoord_index) :
                glm::vec2(0.f)
            );
        }
    }

    std::cout << newObj.meshData.vertices.size() << " vertices loaded" << std::endl;


}

void Scene::buildDevSceneData()
{
    sceneDev = new SceneDev();
    uint32_t triNum = 0;
    uint32_t primNum = 0;

    std::vector<int> mids;

    // calculate prim num and apply transformations
    for (auto& model : objects)
    {
        if (model.type == TRIANGLE)
        {
            uint32_t num = model.meshData.vertices.size() / 3;
            triNum += num;
            primNum += num;
            for (uint32_t i = 0; i < model.meshData.vertices.size(); ++i)
            {
                model.meshData.vertices[i] = glm::vec3(model.transforms.transform * glm::vec4(model.meshData.vertices[i], 1.f));
                model.meshData.normals[i] = glm::normalize(glm::vec3(model.transforms.invTranspose * glm::vec4(model.meshData.normals[i], 1.f)));
            }
            for (uint32_t i = 0; i < num; ++i) mids.push_back(model.materialid);
        }
        else
        {
            geoms.push_back(Geom());
            Geom& tmp = geoms.back();
            tmp.type = model.type;
            tmp.materialid = model.materialid;
            tmp.transforms = model.transforms;
            ++primNum;
        }
    }

    sceneDev->primNum = primNum;
    sceneDev->triNum = triNum;

    // build primitives
    primitives.resize(primNum);
    uint32_t offset = 0;

    // triangle primitives
    for (auto& model : objects)
    {
        if (model.type == TRIANGLE)
        {
            uint32_t num = model.meshData.vertices.size() / 3;
            for (uint32_t i = 0; i < num; ++i)
            {
                primitives[i + offset].primId = i + offset;
                primitives[i + offset].materialId = model.materialid;
                primitives[i + offset].bbox = AABB::getAABB(model.meshData.vertices[3 * i],
                    model.meshData.vertices[3 * i + 1], model.meshData.vertices[3 * i + 2]);
            }
            offset += num;
        }
    }

    // other primitives
    for (uint32_t i = 0; i < geoms.size(); ++i)
    {
        primitives[triNum + i].primId = triNum + i;
        primitives[triNum + i].materialId = geoms[i].materialid;
        primitives[triNum + i].bbox = AABB::getAABB(geoms[i].type, geoms[i].transforms.transform);
    }

    uint32_t treeSize = 0;
    std::vector<AABB> AABBs;
    std::vector<MTBVHNode> flattenNodes = BVH::buildBVH(*this, AABBs, primNum, triNum, treeSize);
    sceneDev->bvhSize = treeSize;

    cudaMalloc(&sceneDev->vertices, 3 * triNum * sizeof(glm::vec3));
    cudaMalloc(&sceneDev->normals, 3 * triNum * sizeof(glm::vec3));
    cudaMalloc(&sceneDev->uvs, 3 * triNum * sizeof(glm::vec2));
    cudaMalloc(&sceneDev->materialIDs, primNum * sizeof(uint32_t));

    cudaMalloc(&sceneDev->primitives, primitives.size() * sizeof(Primitive));
    cudaMemcpy(sceneDev->primitives, primitives.data(), primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->materials, materials.size() * sizeof(Material));
    cudaMemcpy(sceneDev->materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->geoms, geoms.size() * sizeof(Geom));
    cudaMemcpy(sceneDev->geoms, geoms.data(), geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    // copy MTBVH array
    cudaMallocPitch(&sceneDev->bvhNodes, &sceneDev->bvhPitch, treeSize * sizeof(MTBVHNode), 6);
    cudaMemcpy2D(sceneDev->bvhNodes, sceneDev->bvhPitch,
        flattenNodes.data(), treeSize * sizeof(MTBVHNode), treeSize * sizeof(MTBVHNode), 6, cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->bvhAABBs, AABBs.size() * sizeof(AABB));
    cudaMemcpy(sceneDev->bvhAABBs, AABBs.data(), AABBs.size() * sizeof(AABB), cudaMemcpyHostToDevice);


    // copy data
    offset = 0;
    for (const auto& model : objects)
    {
        if (model.type == TRIANGLE)
        {
            uint32_t num = model.meshData.vertices.size();
            cudaMemcpy(sceneDev->vertices + offset, model.meshData.vertices.data(), num * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(sceneDev->normals + offset, model.meshData.normals.data(), num * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(sceneDev->uvs + offset, model.meshData.uvs.data(), num * sizeof(glm::vec2), cudaMemcpyHostToDevice);
            offset += num;
        }
        else
        {
            // push cube and sphere to the end
            mids.push_back(model.materialid);
        }
    }
    cudaMemcpy(sceneDev->materialIDs, mids.data(), primNum * sizeof(int), cudaMemcpyHostToDevice);
}

void SceneDev::freeCudaMemory()
{
    cudaFree(geoms);
    cudaFree(materials);
    cudaFree(materialIDs);
    cudaFree(primitives);
    cudaFree(vertices);
    cudaFree(normals);
    cudaFree(uvs);
}

