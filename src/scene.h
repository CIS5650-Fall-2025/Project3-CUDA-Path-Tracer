#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <stb_image.h>  
#include <stb_image_write.h>  
#include <tiny_gltf.h> 
#include <tiny_obj_loader.h>
#include <unordered_map>


using namespace std;

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName, Geom& newGeom, std::unordered_map<std::string, uint32_t>& MatNameToID);
    void loadTexture(const std::string& texName, Geom& newGeom, std::string path);
    void loadFromGltf(const std::string& gltfName);
public:
    Scene(string filename);
    ~Scene();

    //test
    //Scene();


    std::vector<Geom> geoms;
    std::vector<Material> materials;
    //add for mesh
    std::vector<Triangle> triangles;
    std::vector<Texture> textures;
    RenderState state;
};
