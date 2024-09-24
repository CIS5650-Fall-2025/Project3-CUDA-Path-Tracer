#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

struct BVHNode {
    glm::vec3 aabbMin, aabbMax;          // 24 bytes - aabb can be defined with 6 floats
    unsigned int leftChild;              // 4 bytes - right child idx is always left + 1
    unsigned int firstTriIdx, triCount;  // 8 bytes; total: 36 bytes
    bool isLeaf() { return triCount > 0; }
};

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    std::vector<Triangle> assembleMesh();
    
    std::vector<unsigned int> triangle_indices;
    std::vector<BVHNode> bvhNodes;
    unsigned int rootNodeIdx{ 0 };
    unsigned int nodesUsed{ 1 };
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;

    std::vector<Geom> meshes;
    std::vector<Triangle> mesh_triangles;
    int triangle_count;

    RenderState state;

    void constructBVH();
    void intersectBVH(Ray& ray, const unsigned int nodeIdx); 
    void updateNodeBounds(unsigned int nodeIdx);
    void subdivide(unsigned int noideIdx);
    bool intersectAABB(const Ray& ray, const glm::vec3 bmin, const glm::vec3 bmax);
};
