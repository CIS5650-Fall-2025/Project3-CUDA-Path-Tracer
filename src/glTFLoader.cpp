#include "glTFLoader.h"

#ifndef TINYGLTF_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#endif
#include <tiny_gltf.h>
#include <iostream>

bool glTFLoader::loadModel(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        printf("Warning: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Error: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to load glTF file: %s\n", filename.c_str());
        return false;
    }

    processNodes(model);
    loadImages(model);
    return true;
}


void glTFLoader::processNodes(const tinygltf::Model& model)
{
    for (const auto& scene : model.scenes) {
        for (int nodeIndex : scene.nodes) {
            traverseNode(model, nodeIndex, glm::mat4(1.0f));
        }
    }
}

void glTFLoader::traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentTransform) {
    const tinygltf::Node& node = model.nodes[nodeIndex];

    glm::mat4 localTransform = getNodeTransform(node);
    glm::mat4 globalTransform = parentTransform * localTransform;

    if (node.mesh >= 0) {
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (const auto& primitive : mesh.primitives) {
            std::cout << "primName: " << mesh.name << "\n";
            extractWorldSpaceTriangleBuffers(model, primitive, globalTransform);
            //primNum++;
        }
    }

    for (int childIndex : node.children) {
        traverseNode(model, childIndex, globalTransform);
    }
}

glm::mat4 glTFLoader::getNodeTransform(const tinygltf::Node& node) {
    glm::mat4 translation(1.0f);
    glm::mat4 rotation(1.0f);
    glm::mat4 scale(1.0f);

    if (node.translation.size() == 3) {
        translation = glm::translate(glm::mat4(1.0f), glm::vec3(
            node.translation[0], node.translation[1], node.translation[2]));
    }

    if (node.rotation.size() == 4) {
        glm::quat q(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
        rotation = glm::mat4_cast(q);
    }

    if (node.scale.size() == 3) {
        scale = glm::scale(glm::mat4(1.0f), glm::vec3(
            node.scale[0], node.scale[1], node.scale[2]));
    }

    if (node.matrix.size() == 16) {
        return glm::make_mat4(node.matrix.data());
    }

    return translation * rotation * scale;
}

void glTFLoader::loadImages(const tinygltf::Model& model)
{
    images.clear();
    for (const auto& texture : model.textures) {
        const tinygltf::Image& image = model.images[texture.source];
        images.push_back(image);
    }
    //std::cout << "Image count = " << images.size() << "\n";
    //std::cout << "size of tinygltf image = " << sizeof(tinygltf::Image) << "\n";
}

void glTFLoader::extractWorldSpaceTriangleBuffers(const tinygltf::Model& model, const tinygltf::Primitive& primitive, const glm::mat4& transform)
{
    if (primitive.attributes.find("POSITION") == primitive.attributes.end()) {
        return;
    }
    Mesh newMesh;
    // Extract position data
    if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
        const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("POSITION")];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

        const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
        newMesh.positions.assign(positions, positions + accessor.count * 3);
    }

    // Extract index data
    if (primitive.indices >= 0) {
        const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

        newMesh.indices.reserve(accessor.count);

        switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            const uint16_t* indices = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            for (size_t i = 0; i < accessor.count; ++i) {
                newMesh.indices.push_back(static_cast<uint32_t>(indices[i]));
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            const uint32_t* indices = reinterpret_cast<const uint32_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            newMesh.indices.assign(indices, indices + accessor.count);
            break;
        }
        default:
            printf("Unsupported index component type\n");
            break;
        }
    }

    //every group of 3 floats needs to be multiplied by transform and then updated!
    for (int i = 0; i < newMesh.positions.size() / 3; i++) {
        float idx = i * 3;
        float x = newMesh.positions[idx];
        float y = newMesh.positions[idx + 1];
        float z = newMesh.positions[idx + 2];
        glm::vec4 v = glm::vec4(x, y, z, 1);
        v = transform * v;
        newMesh.positions[idx] = v.x;
        newMesh.positions[idx + 1] = v.y;
        newMesh.positions[idx + 2] = v.z;
    }


    //UVS
    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
        //const auto& uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
        //const auto& uvBufferView = model.bufferViews[uvAccessor.bufferView];
        //const auto& uvBuffer = model.buffers[uvBufferView.buffer];

        //// Load UVs
        //const float* uvs = reinterpret_cast<const float*>(&uvBuffer.data[uvBufferView.byteOffset]);
        //newMesh.uvs.assign(uvs, uvs + uvAccessor.count * 2);

        int texcoordAccessorIdx = primitive.attributes.find("TEXCOORD_0")->second;
        const tinygltf::Accessor& texcoordAccessor = model.accessors[texcoordAccessorIdx];
        const tinygltf::BufferView& bufferView = model.bufferViews[texcoordAccessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

        // Access the raw UV data from the buffer
        const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + texcoordAccessor.byteOffset;

        // Extract the UVs based on the component type (typically float)
        //std::vector<float> uvs;
        if (texcoordAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
            const float* floatData = reinterpret_cast<const float*>(dataPtr);

            for (size_t i = 0; i < texcoordAccessor.count; i++) {
                float u = floatData[i * 2 + 0]; // U coordinate
                float v = floatData[i * 2 + 1]; // V coordinate

                newMesh.uvs.push_back(u);
                newMesh.uvs.push_back(v);

                // Optionally print the UVs for debugging
                //std::cout << "UV: (" << u << ", " << v << ")" << std::endl;
            }
        }
    }
    else {
        std::cout << "Primitive does not have UVs." << std::endl;
    }

    
    int materialIndex = primitive.material;
    //std::cout << "materialIndex: " << materialIndex << "\n";
    if (materialIndex != -1) {
        const auto& material = model.materials[materialIndex];
        if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            for (int i = 0; i < newMesh.positions.size() / 3; i++) {
                newMesh.baseColorTextureIDs.push_back(material.pbrMetallicRoughness.baseColorTexture.index);
            }
        }
        for (int i = 0; i < newMesh.positions.size() / 3; i++) {
            newMesh.matIDs.push_back(materialIndex);
        }
    }

    meshes.push_back(newMesh);
}

AABB glTFLoader::calculateBounds(int start, int end)
{
    AABB bounds;
    bounds.min = glm::vec3(FLT_MAX);
    bounds.max = glm::vec3(-FLT_MAX);

    for (int i = start; i < end; i++) {
        const MeshTriangle& tri = (*triangles)[BVHtriangleIndexBuffer[i]];
        bounds.min = min(bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
        bounds.max = max(bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
    }

    return bounds;
}

int glTFLoader::buildBVHRecursive(int start, int end, int depth) {
    nodesUsed++;
    int nodeIndex = nodesUsed;

    AABB bounds = calculateBounds(start, end);
    nodes[nodeIndex].bounds = bounds;

    int triangleCount = end - start;

    if (triangleCount <= 4) {
        // Leaf node
        //std::cout << "building leaf node!\n";
        nodes[nodeIndex].leftChild = -1;
        nodes[nodeIndex].rightChild = -1;
        //nodes[nodeIndex].firstTriangle = start;
        //nodes[nodeIndex].triangleCount = triangleCount;

        for (int i = 0; i < triangleCount; i++) {
            nodes[nodeIndex].triangleIDs[i] = BVHtriangleIndexBuffer[start + i];
            //usually, it would just be triangle[start + i] to get the tri
            //or in this case, it would just be start + i
            //but start + i is not the correct index anymore
        }
    }
    else {
        // Internal node
        int axis = longestAxis(bounds);
        int mid = (start + end) / 2;

        std::sort(BVHtriangleIndexBuffer.begin() + start, BVHtriangleIndexBuffer.begin() + end, CompareTriangles(axis, triangles.get()));

        nodes[nodeIndex].leftChild = buildBVHRecursive(start, mid, depth + 1);
        nodes[nodeIndex].rightChild = buildBVHRecursive(mid, end, depth + 1);
        nodes[nodeIndex].triangleIDs = glm::ivec4(-1, -1, -1, -1);
    }

    return nodeIndex;
}

int glTFLoader::longestAxis(const AABB& bounds) {
    glm::vec3 extent = bounds.max - bounds.min;
    if (extent.x > extent.y && extent.x > extent.z) return 0;
    if (extent.y > extent.z) return 1;
    return 2;
}