#include "mesh.h"
#include "tiny_gltf.h"
#include <string>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp> 
#include <iostream>

void mesh::loadGLTF(std::string filename) {


}

tinygltf::Model LoadModel(const std::string& filepath) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;

    bool success = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
    if (!success) {
        std::cerr << "Failed to load glTF model: " << err << std::endl;
    }
    if (!warn.empty()) {
        std::cout << "Warning: " << warn << std::endl;
    }

    return model;
}