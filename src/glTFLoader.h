#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_gltf.h>
#include <iostream>

struct MeshTriangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;


    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;  // UV coordinates for each vertex of the triangle

    int textureIdx;
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
    };

    glTFLoader() {}

    bool loadModel(const std::string& filename);

    std::vector<MeshTriangle> getTriangles() const {
        std::vector<MeshTriangle> triangles;
        for (const auto& mesh : meshes) {
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                MeshTriangle tri;
                tri.v0 = getVertex(mesh, mesh.indices[i]);
                tri.v1 = getVertex(mesh, mesh.indices[i + 1]);
                tri.v2 = getVertex(mesh, mesh.indices[i + 2]);
                //tri.uv
                tri.uv0 = getUV(mesh, mesh.indices[i]);
                tri.uv1 = getUV(mesh, mesh.indices[i + 1]);
                tri.uv2 = getUV(mesh, mesh.indices[i + 2]);

                //tri.uv0
                triangles.push_back(tri);
            }
        }
        std::cout << "# of triangles: "<< triangles.size() <<"\n";
        return triangles;
    }

    std::vector<tinygltf::Image> getImages() const {
        return images;
    }

private:
    std::vector<Mesh> meshes;
    std::vector<tinygltf::Image> images;

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
};


















//#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION

/*
* Use tinyGltf to load .gltf mesh data including triangles and (eventually) texture data!
* 
* Traverse the gltf file's nodes, meshes, and primitives to access position data: vertex positions and index buffers
*		Also, need to account for node' transformations
* 
* GLTF->node->mesh->primitives->buffers!
* GLTF->node->mesh->primitives->materials! (including textures!)
* GLTF->node->mesh->primitives->texcoords(UVs)
* 
* 
* Triangle positions are not in world-space, they are in node-space. Need to apply node-transformations to bring them all to world-space!
* We want all triangles to be in world-space! (this will be useful for BVH optimization afterwards.)
*/

//class glTFLoader {
//private:
//		bool modelLoaded = false;
//		tinygltf::Model model;
//		tinygltf::TinyGLTF loader;
//		std::string err;
//		std::string warn;
//public:
//		bool loadModel(const std::string pathToFile);
//		void extractPositions();
//		glm::mat4 getGlobalTransform(const tinygltf::Node& node);
//		//void extractModelData();
//		//void extractMeshFromNode(const int i, const glm::mat4 parentTransform);
//		//void extractTrianglesFromMesh(const int meshID, const glm::mat4 transform);
//};