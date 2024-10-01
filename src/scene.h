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
    void InitializeCameraAndRenderState();
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName);
public:
    Scene();
    ~Scene();
    void LoadFromFile(string filename);
    bool sceneReady = false;
    bool useMesh = false;

    Mesh mesh;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
