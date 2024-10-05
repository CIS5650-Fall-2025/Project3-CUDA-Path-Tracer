#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void loadFromObj(std::string path, int idx, Geom& geom);
    int loadTexture(std::string path, std::string name);
    
    void buildBVH(int maxDepth);
    void updateNodeBounds(int nodeIdx);
    void subdivide(int nodeIdx, int currDepth, int maxDepth);

    float splitPlane(BVHNode& node, int& bestAxis, float& splitPos);

    void Scene::buildAABB(Geom& geom);


public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<Texture> textures;

    std::vector<BVHNode> bvhNode;
    int rootNodeIdx = 0;
    int nodesUsed = 1;

    RenderState state;
};