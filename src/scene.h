#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include <unordered_map>
#include "utilities.h"
#include "sceneStructs.h"
#include "BVH.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    int loadFromObj(const std::string& filePath, bool withinJsonFile, std::unordered_map<std::string, uint32_t>& MatNameToID);
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<BVHNode> bvhNodes;
    int bvhRootIndex;
    std::vector<int> geomIdx;
};
