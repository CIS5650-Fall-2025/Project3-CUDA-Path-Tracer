#pragma once

#ifndef SCENE_H
#define SCENE_H

#include <json.hpp>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "Bounds3.hpp"
#include "cudautils.hpp"
#include "BVH.h";
#include "tiny_obj_loader.hpp"
#include "material.h"
#include "image.h"
#include "lightSample.h"
#include "distribution1D.h"
#include "distribution2D.h"

#include <gltf/tiny_gltf.h>

using namespace std;

//We have to transit the data of scene in cpu to the devscene in gpu
class Scene; 

class DevScene
{
public:
    void initiate(Scene& scene);
    void destroy();

    int tri_num;
    int bvh_size;
    Triangle* dev_triangles;
    GpuBVHNode* dev_gpuBVH;

    int tex_num;
    DevTexObj* dev_textures;
    glm::vec3* dev_texture_data;

    int envMapID = -1;
    DevTexSampler dev_envSampler{0.f};

    lightPrim* dev_lights;

    LightSampler dev_lightSampler;
    DevDistribution1D dev_envDistribution1D;
    DevDistribution2D dev_envDistribution2D;

    Geom* dev_geoms;
    Material* dev_materials;
};

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadTexture(const string& fileName, float gamma = 1.f);

public:
    Scene(const string& filename);
    void setDevData();
    void clear();

    int loadGltfTexture(const tinygltf::Image& image, const string& gltfPath);

    std::vector<Geom> geoms;
    std::map<string, int> geomNameMap;
    std::vector<GPUGeom> gpuGeoms;
    std::vector<Material> materials;
    std::map<string, int> materialNameMap;
    std::vector<Triangle> triangles;
    std::vector<image*> textures;
    std::map<image*, int> textureIdMap;
    std::vector<lightPrim> lights;

    int envMapID = -1;

    std::vector<GpuBVHNodeInfo> gpuBVHNodeInfos;
    std::vector<GpuBVHNode> gpuBVHNodes;
    RecursiveBVHNode* bvhRoot;
    Distribution1D envDistribution1D;
    Distribution2D envDistribution2D;

    BVHAccel bvhConstructor;
    DevScene tempDevScene;
    DevScene* dev_scene;
    RenderState state;
};


class MeshData;

namespace Resource
{
    std::vector<MeshData*> loadObj(const string& filename, const int _geomIdx);
	std::vector<MeshData*> loadGLTF(const string& filename, const int _geomIdx, Scene& scene, std::vector<int>& sceneMaterialList);
    void clear();
    image* loadTexture(const std::string& filename, float gamma = 1.f);

    extern int meshCount;
    extern std::map<std::string, std::vector<MeshData*>> meshPool;
    extern std::map<std::string, image*> texturePool;
}

//Meshdata contains data of an obj or other 3d model file, it is prototype of Geometry, area and bounding box is used to calculate BVH tree and intersections.
class MeshData
{
public:
    //float area;
    //Bounds3 boundingBox;
    std::vector<Triangle> triangles;
};

#endif // SCENE_H