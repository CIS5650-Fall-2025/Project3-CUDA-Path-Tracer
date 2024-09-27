#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;



struct AABB 
{
    glm::vec3 min;
    glm::vec3 max;

    // Default constructor initializes min to the largest possible value and max to the smallest.
    __host__ __device__ AABB() : min(glm::vec3(std::numeric_limits<float>::max())), max(glm::vec3(std::numeric_limits<float>::lowest())) {}

    // Constructor that initializes the AABB with given min and max points.
    __host__ __device__ AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

    // Constructor that initializes the AABB to enclose three points.
    __host__ __device__ AABB(const glm::vec3& point1, const glm::vec3& point2, const glm::vec3& point3) {
        min = glm::min(point1, glm::min(point2, point3));
        max = glm::max(point1, glm::max(point2, point3));
    }

    // Method to find the index of the longest axis (0: x, 1: y, 2: z).
    __host__ __device__ int LongestAxisIndex() const {
        glm::vec3 diagonal = (max - min);
        int maxAxis = (diagonal.x > diagonal.y) ? 0 : 1;
        return (diagonal[maxAxis] > diagonal.z) ? maxAxis : 2;
    }

    // Method to compute the surface area of the AABB.
    __host__ __device__ float SurfaceArea() const {
        glm::vec3 d = (max - min);
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};




struct Triangle
{
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];

    AABB aabb = AABB();
};



struct BVH_Node
{
    AABB aabb;       // The axis-aligned bounding box that encloses this node.
    int leftChild = -1;     // Index of the left child node.
    int rightChild = -1;    // Index of the right child node.

    int splitAxis = -1;     // The axis along which the node is split (0: x, 1: y, 2: z).
    int fisrtT_idx = -1; // the first triangle in the leaf node.
    int TCount = -1;       // Number of traingles contained in the node.
};


struct BVH_TriangleAABB_relation
{
    int triangle_id;
    AABB aabb;

    __host__ __device__ BVH_TriangleAABB_relation() : triangle_id(-1), aabb(AABB()) {}

    __host__ __device__ BVH_TriangleAABB_relation(int id, const AABB& aabb_in) : triangle_id(id), aabb(aabb_in) {}
};




static AABB combine2AABB(const AABB& aabb1, const AABB& aabb2) {
    return AABB(
        glm::min(aabb1.min, aabb2.min),  // Get the minimum point for the new bounding box
        glm::max(aabb1.max, aabb2.max)   // Get the maximum point for the new bounding box
    );
}



class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;


    std::vector<Triangle> BVH_triangles;
    std::vector<BVH_Node> BVH_Nodes;


    int constructBVH(int, int);
    void updateBVH(const std::vector<BVH_TriangleAABB_relation>&, std::vector<BVH_Node>&, int);

    int Scene::subdivide(std::vector<BVH_TriangleAABB_relation>&, std::vector<BVH_Node>&, int, int&, int&);

    void loadCamera();
};
