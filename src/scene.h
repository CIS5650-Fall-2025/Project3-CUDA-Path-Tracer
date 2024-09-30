#pragma once

#include <vector>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <iostream>
#include "tiny_obj_loader.h"
#include "sceneDev.h"

//using namespace std;

class Scene
{
private:
    std::string sceneFileName;
    void loadFromJSON(const std::string& jsonName);
    bool loadObj(const std::string& objPath);
    void loadObjMaterials(const std::string& mtlPath, std::vector<tinyobj::material_t>& mats);
    void loadTextureFile(const std::string& texPath, cudaTextureObject_t& texObj);
    void loadEnvMap(const std::string& texPath);
    void createCudaTexture(void* data, int width, int height, cudaTextureObject_t& texObj, bool isHDR);

    void scatterPrimitives(std::vector<Primitive>& srcPrim, std::vector<PrimitiveDev>& dstPrim,
        std::vector<glm::vec3>& dstVec,
        std::vector<glm::vec3>& dstNor,
        std::vector <glm::vec2>& dstUV);

public:
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Object> objects;
    std::unordered_map<std::string, uint32_t> MatNameToID;
    std::unordered_map<std::string, cudaTextureObject_t> TextureNameToID;
    std::string skyboxPath;

    // full scene primitives
    std::vector<Primitive> primitives;
    //std::vector<glm::vec3> vertices;
    //std::vector<glm::vec3> normals;
    //std::vector<glm::vec3> uvs;
    RenderState state;
    SceneDev* sceneDev;

    void loadSceneModels();
    void buildDevSceneData();
};