#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "bvh.h"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Material> materials;

    std::vector<Geom> geoms;
    BVH bvh;
    std::vector<BVH::Node> nodes;
    std::vector<glm::vec4> textures;
    
    RenderState state;
};
