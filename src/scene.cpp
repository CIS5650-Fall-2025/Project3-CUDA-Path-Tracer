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
        loadFromGLTF(filename);
        return;
    }
    else if (ext == ".gltf")
    {
        loadFromGLTF(filename);
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
            newMaterial.specular.color = newMaterial.color;
            const float& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - roughness;
            newMaterial.hasRefractive = 0.0f;
            newMaterial.emittance = 0.0f;
        }
        else if (p["TYPE"] == "Dielectric") {
            newMaterial.color = glm::vec3(p["RGB"][0], p["RGB"][1], p["RGB"][2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasReflective = 0.0f;
            newMaterial.emittance = 0.0f;
            newMaterial.specular.color = glm::vec3(1, 1, 1);
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

void Scene::LoadMaterialsFromFromGLTF() {
  meshesMaterials = gltfLoader->LoadMaterials(meshesTextures);
}

void Scene::loadFromGLTF(const std::string& gltfFilename)
{
    // gltfLoader = std::make_unique<GLTFLoader>(string("C:/Users/carlo/Code/carlos-lopez-garces/d3d12/Assets/BoomBox/BoomBox.gltf"));
    gltfLoader = std::make_unique<GLTFLoader>(string("C:/Users/carlo/Code/carlos-lopez-garces/CIS5650/Penn-CIS-5650-Project3-CUDA-Path-Tracer/scenes/DamagedHelmet/DamagedHelmet.gltf"));
    // gltfLoader = std::make_unique<GLTFLoader>(string("C:/Users/carlo/Code/carlos-lopez-garces/d3d12/Assets/Sponza/Sponza.gltf"));
    // gltfLoader = std::make_unique<GLTFLoader>(string("C:/Users/carlo/Code/carlos-lopez-garces/CIS5650/Penn-CIS-5650-Project3-CUDA-Path-Tracer/scenes/FlightHelmet/FlightHelmet.gltf"));
    gltfLoader->LoadModel();
    // LoadTexturesFromGLTF();
    // LoadMaterialsFromFromGLTF();
    LoadGeometryFromGLTF();
}

__host__ __device__ AABB unionAABB(const AABB& a, const AABB& b) {
    AABB result;
    result.min = glm::min(a.min, b.min);  // Take the minimum of both AABBs' mins
    result.max = glm::max(a.max, b.max);  // Take the maximum of both AABBs' maxs
    return result;
}

void buildBVH(
    std::vector<BVHNode>& bvhNodes,  // Output: the list of BVH nodes
    int nodeIdx,                     // The current node index in the BVH
    int* objectIndices,              // The list of object (triangle/mesh) indices
    int start,                       // Start index of the current partition
    int end,                         // End index of the current partition
    const std::vector<AABB>& objectBounds  // The bounding boxes of the objects
) {
    // Access the current node by index instead of using a reference
    bvhNodes[nodeIdx].bounds = objectBounds[objectIndices[start]];
    for (int i = start + 1; i < end; ++i) {
        bvhNodes[nodeIdx].bounds = unionAABB(bvhNodes[nodeIdx].bounds, objectBounds[objectIndices[i]]);
    }

    int numObjects = end - start;

    // If this is a leaf node, store the object range and exit recursion
    if (numObjects <= 3) {
        bvhNodes[nodeIdx].isLeaf = true;
        bvhNodes[nodeIdx].start = start;
        bvhNodes[nodeIdx].end = end;
        return;
    }

    // Choose the axis to split along (largest axis of the bounding box)
    glm::vec3 extent = bvhNodes[nodeIdx].bounds.max - bvhNodes[nodeIdx].bounds.min;
    int axis = (extent.x > extent.y) ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);

    // Sort the object indices based on the center of their bounding box along the chosen axis
    std::sort(objectIndices + start, objectIndices + end, [&](int a, int b) {
        return (objectBounds[a].min[axis] + objectBounds[a].max[axis]) / 2.0f <
               (objectBounds[b].min[axis] + objectBounds[b].max[axis]) / 2.0f;
    });

    // Split the objects into two equal parts
    int mid = (start + end) / 2;

    // Create child nodes by using the size of the current vector before emplacing
    bvhNodes[nodeIdx].isLeaf = false;
    
    // Left child
    int leftChildIdx = bvhNodes.size();
    bvhNodes.emplace_back();  // Add a new BVHNode
    bvhNodes[nodeIdx].left = leftChildIdx;
    buildBVH(bvhNodes, leftChildIdx, objectIndices, start, mid, objectBounds);

    // Right child
    int rightChildIdx = bvhNodes.size();
    bvhNodes.emplace_back();  // Add a new BVHNode
    bvhNodes[nodeIdx].right = rightChildIdx;
    buildBVH(bvhNodes, rightChildIdx, objectIndices, mid, end, objectBounds);
}

void Scene::LoadGeometryFromGLTF() {
    unsigned int nodeCount = gltfLoader->getNodeCount();

    Material newMaterial{};
    newMaterial.color = glm::vec3(1, 1, 1);
    newMaterial.emittance = 5.0;
    materials.emplace_back(newMaterial);

    // std::vector<AABB> meshBounds;  // Bounding boxes of all meshes
    // std::vector<int> meshIndices;  // Mesh indices for the top-level BVH
    // int meshIdx = 0;

    for (int nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
        unsigned int primCount = gltfLoader->getPrimitiveCount(nodeIdx);

        for (int primIdx = 0; primIdx < primCount; ++primIdx) {
            Geom newGeom;
            newGeom.type = MESH;

            GLTFPrimitiveData gltfData = gltfLoader->LoadPrimitive(nodeIdx, primIdx);

            newGeom.offset = meshesPositions.size();
            newGeom.count = gltfData.positions.size();
            meshesPositions.insert(meshesPositions.end(), gltfData.positions.begin(), gltfData.positions.end());

            newGeom.indexOffset = meshesIndices.size();
            newGeom.indexCount = gltfData.indices.size();
            meshesIndices.insert(meshesIndices.end(), gltfData.indices.begin(), gltfData.indices.end());

            if (!gltfData.normals.empty()) {
                meshesNormals.insert(meshesNormals.end(), gltfData.normals.begin(), gltfData.normals.end());
            }

            if (!gltfData.uvs.empty()) {
                meshesUVs.insert(meshesUVs.end(), gltfData.uvs.begin(), gltfData.uvs.end());
            }

            newGeom.materialid = materials.size() - 1;
            // newGeom.triangleCount = gltfData.indices.size() / 3;

            newGeom.transform = utilityCore::buildTransformationMatrix(
                glm::vec3(0, 5, 0), glm::vec3(0, 0, 0), glm::vec3(5, 5, 5)
            );

            // // Create triangleIndices for the mesh (in host memory)
            // std::vector<int> triangleIndices(newGeom.triangleCount);
            // for (size_t i = 0; i < newGeom.triangleCount; ++i) {
            //     triangleIndices[i] = i;  // Index of each triangle
            // }

            // // Compute bounding box for the mesh
            // AABB meshAABB;
            // for (int i = 0; i < newGeom.count; ++i) {
            //     glm::vec3 vertex = meshesPositions[newGeom.offset + i];
            //     glm::vec3 transformedVertex = glm::vec3(newGeom.transform * glm::vec4(vertex, 1.0f));
            //     meshAABB.min = glm::min(meshAABB.min, transformedVertex);
            //     meshAABB.max = glm::max(meshAABB.max, transformedVertex);
            // }

            // // Build per-mesh BVH using triangle bounds
            // std::vector<AABB> triangleBounds(newGeom.triangleCount);
            // for (size_t i = 0; i < newGeom.triangleCount; ++i) {
            //     int idx = newGeom.indexOffset + i * 3;
            //     glm::vec3 v0 = gltfData.positions[gltfData.indices[idx]];
            //     glm::vec3 v1 = gltfData.positions[gltfData.indices[idx + 1]];
            //     glm::vec3 v2 = gltfData.positions[gltfData.indices[idx + 2]];

            //     AABB triangleAABB;
            //     triangleAABB.min = glm::min(v0, glm::min(v1, v2));
            //     triangleAABB.max = glm::max(v0, glm::max(v1, v2));
            //     triangleBounds[i] = triangleAABB;
            // }

            // // Build the per-mesh BVH in host memory
            // std::vector<BVHNode> meshBVH;
            // meshBVH.emplace_back();  // Root node
            // buildBVH(meshBVH, 0, triangleIndices.data(), 0, triangleIndices.size(), triangleBounds);

            // // Store the BVH and triangle indices in the Geom structure for later use
            // newGeom.triangleIndices = new int[newGeom.triangleCount];  // Allocate memory on the host
            // std::memcpy(newGeom.triangleIndices, triangleIndices.data(), newGeom.triangleCount * sizeof(int));  // Copy triangle indices to host memory

            // newGeom.meshBVH = new BVHNode[meshBVH.size()];  // Allocate memory for BVH on the host
            // newGeom.meshBVHCount = meshBVH.size();
            // std::memcpy(newGeom.meshBVH, meshBVH.data(), meshBVH.size() * sizeof(BVHNode));  // Copy BVH to host memory

            // // Set the BVH root for this mesh
            // newGeom.bvhRoot = 0;

            // // Store the mesh bounding box for the top-level BVH
            // meshBounds.push_back(meshAABB);
            // meshIndices.push_back(meshIdx++);

            geoms.push_back(newGeom);
        }
    }

    // // Build the top-level BVH for all meshes
    // std::vector<BVHNode> topLevelBVHNodes;
    // topLevelBVHNodes.emplace_back();  // Add the root node
    // buildBVH(topLevelBVHNodes, 0, meshIndices.data(), 0, meshIndices.size(), meshBounds);

    // // Store the top-level BVH in host memory
    // topLevelBVH = new BVHNode[topLevelBVHNodes.size()];  // Allocate memory for the top-level BVH on the host
    // topLevelBVHCount = topLevelBVHNodes.size();
    // std::memcpy(topLevelBVH, topLevelBVHNodes.data(), topLevelBVHNodes.size() * sizeof(BVHNode));  // Copy top-level BVH to host memory
}
