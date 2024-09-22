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

    // declare the vector that stores all the bounding sphere generation data
    std::vector<bounding_sphere_generation_data> bounding_sphere_generations {};

    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    // declare the vector that stores all the vertices
    std::vector<vertex_data> vertices {};

    // declare the vector that stores all the bounding sphere data
    std::vector<bounding_sphere_data> bounding_spheres {};

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
