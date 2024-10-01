#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "glm/glm.hpp"
#include "utilities.h"
#include "bvh.h"
#include "sceneStructs.h"
using namespace std;

struct bbox;
struct bvhNode;

class Scene
{
private:
    ifstream fp_in;
    void InitializeCameraAndRenderState();
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName);
    void transformToTarget(const glm::vec3& bboxCenter, const glm::vec3& bboxScale);
public:
    Scene();
    ~Scene();
    void LoadFromFile(string filename);
    void autoCentralize();
    bool sceneReady = false;

    Mesh mesh;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::unordered_map<std::string, uint32_t> MatNameToID;

    // bvh
    std::vector<bbox> triangleBboxes;
    std::vector<bvhNode> bvhNodes;

    void buildBVH();

    // options
    bool renderWithPathTracing = true;
    bool sortByMaterial = false;
    bool autoCentralizeObj = true;
};
