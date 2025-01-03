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
    template <typename T, typename U>
    std::vector<U> castBufferToVector(const unsigned char* buffer, size_t count, int stride);
    void loadGLTF(const std::string& filename, int material, const glm::vec3 translation, const glm::vec3 rotation, const glm::vec3 scale);

public:
    Scene(string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<Texture> textures;
    RenderState state;
};
