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

    void expandBounds(int start, int end, glm::vec3& mins, glm::vec3& maxs);
    int getSubtreeSize(int nodeIndex);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<BVHNode> nodes;
    std::vector<int> indices;
    void constructBVHTree();
    void buildBVHTree(int start, int end);
};
