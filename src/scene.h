#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "gltf/tiny_gltf.h"
#include "gltf/stb_image.h"
#include <tiny_obj_loader.h>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void loadObj(Geom& newGeom, string obj_filename, string scene_filename);
    int buildBVHEqualCount(int meshStartIdx, int meshEndIdx);
    int loadGltf(Geom& newGeom, string filename, string scene_filename);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    //Obj Mesh, currently only one obj per scene is supported 
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<Mesh> meshes;
    std::vector<BVHNode> bvh;
};
