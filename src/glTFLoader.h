#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_gltf.h>
#include <iostream>

using uint = unsigned int;

struct AABB {
    glm::vec3 min, max;
};

struct BVHNode {
    AABB bounds;
    int leftChild;
    int rightChild;
    int firstTriangle;
    int triangleCount;
};

struct MeshTriangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;


    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;  // UV coordinates for each vertex of the triangle

    int baseColorTexID;
    int materialIndex; //right now... we are just relying on the json single material for the whole mesh, but we should do it relative to the unique 
                       //materials within a gltf file! Eventually, a single scene shld be able to have its own unique materials inside, not just one.
                       //What I might do is produce a mapping inside the json file that maps gltf material indices to real defined materials.
};

class glTFLoader {
public:

    struct Mesh {
        std::vector<float> positions; // Stores x, y, z coordinates consecutively
        std::vector<uint32_t> indices;
        std::vector<float> uvs;
        std::vector<int> baseColorTextureIDs;
        std::vector<int> matIDs;
    };

    glTFLoader() {}

    bool loadModel(const std::string& filename);
    void extractTriangles() {
        triangles = std::make_unique<std::vector<MeshTriangle>>();
        for (const auto& mesh : meshes) {
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                MeshTriangle tri;
                tri.v0 = getVertex(mesh, mesh.indices[i]);
                tri.v1 = getVertex(mesh, mesh.indices[i + 1]);
                tri.v2 = getVertex(mesh, mesh.indices[i + 2]);

                tri.uv0 = getUV(mesh, mesh.indices[i]);
                tri.uv1 = getUV(mesh, mesh.indices[i + 1]);
                tri.uv2 = getUV(mesh, mesh.indices[i + 2]);

                std::cout << "( " << tri.uv0.x << ", " << tri.uv0.y << " )\n";

                if (mesh.baseColorTextureIDs.size() > 0) {
                    tri.baseColorTexID = getBaseColorTextureID(mesh, mesh.indices[i]);
                }
                else {
                    tri.baseColorTexID = -1;
                }
                if (mesh.matIDs.size() > 0) {
                    tri.materialIndex = getMatID(mesh, mesh.indices[i]);
                }

                //Let's also set materialIdx!

                triangles->push_back(tri);
            }
        }
    }

    std::vector<MeshTriangle>* getTriangles() {
        if (triangles == nullptr) {
            extractTriangles();
            //std::cout << "PT0: TRIANGLE BUFFER INITIALIZED, GLTF SPACE, NOT WORLD SPACE YET.\n";
        }
        return triangles.get();
    }

    std::vector<tinygltf::Image> getImages() const {
        return images;
    }

    //get BVH
    std::vector<BVHNode> getBVHTree() {
        if (nodes.size() == 0) {
            buildBVH();
        }
        return nodes;
    }

private:
    unsigned int rootNodeIdx = 0;
    unsigned int nodesUsed = -1;
    std::vector<Mesh> meshes;
    std::vector<tinygltf::Image> images;
    std::vector<BVHNode> nodes;
    std::unique_ptr<std::vector<MeshTriangle>> triangles;
    std::vector<unsigned int> triIdx;

    void processNodes(const tinygltf::Model& model);
    void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentTransform);
    glm::mat4 getNodeTransform(const tinygltf::Node& node);
    void loadImages(const tinygltf::Model& model);
    void extractWorldSpaceTriangleBuffers(const tinygltf::Model& model, const tinygltf::Primitive& primitive, const glm::mat4& transform);

    glm::vec3 getVertex(const Mesh& mesh, uint32_t index) const {
        size_t i = index * 3;
        return { mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2] };
    }

    glm::vec2 getUV(const Mesh& mesh, uint32_t index) const {
        size_t i = index * 2;
        return { mesh.uvs[i], mesh.uvs[i + 1] };
    }
    int getBaseColorTextureID(const Mesh& mesh, uint32_t index) const {
        return mesh.baseColorTextureIDs[index];
    }

    int getMatID(const Mesh& mesh, uint32_t index) const {
        return mesh.matIDs[index];
    }


    //BVH

    AABB calculateBounds(int start, int end);
    int buildBVHRecursive(int start, int end, int depth);

    void buildBVH() {
        //std::cout << "AHHHHHHHHHHHHHH!\n";
        nodes.clear();
        nodes.resize(triangles->size());
        //std::cout << "calling rec. start is 0 and end is " << triangles->size() << "\n";
        rootNodeIdx = buildBVHRecursive(0, triangles->size(), 0);
        //std::cout << "PART 2: THE TRIANGLE BUFFER HAS BEEN MODIFIED DUE TO BVH CREATION" << "\n";
    }

    int longestAxis(const AABB& bounds);

    glm::vec3 centroid(const MeshTriangle& tri) {
        return (tri.v0 + tri.v1 + tri.v2) * 0.3333f;
    }
};