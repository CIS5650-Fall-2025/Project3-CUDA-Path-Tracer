#include "mesh.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"
#include <string>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp> 
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "mesh.h"


using namespace std;

#if 0
void mesh::loadOBJ(std::string filename){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

    if (!warn.empty()) {
        std::cout << "WARNING: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
    }
    if (!ret) {
        exit(1);
    }

    // Now convert OBJ data to Geom structure
    for (const auto& shape : shapes) {
        Geom meshGeom;
        meshGeom.type = MESH;
        meshGeom.numTriangles = shape.mesh.indices.size() / 3;
        meshGeom.triangles = new Triangle[meshGeom.numTriangles];

        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            int idx0 = shape.mesh.indices[i].vertex_index;
            int idx1 = shape.mesh.indices[i + 1].vertex_index;
            int idx2 = shape.mesh.indices[i + 2].vertex_index;

            glm::vec3 v0(attrib.vertices[3 * idx0], attrib.vertices[3 * idx0 + 1], attrib.vertices[3 * idx0 + 2]);
            glm::vec3 v1(attrib.vertices[3 * idx1], attrib.vertices[3 * idx1 + 1], attrib.vertices[3 * idx1 + 2]);
            glm::vec3 v2(attrib.vertices[3 * idx2], attrib.vertices[3 * idx2 + 1], attrib.vertices[3 * idx2 + 2]);

            meshGeom.triangles[i / 3] = { v0, v1, v2 };
        }

        geoms.push_back(meshGeom);
    }
}

#endif
void mesh::loadGLTF(std::string filename) {

}


tinygltf::Model mesh::LoadModel(std::string& filepath) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    //ascii file
    //bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, argv[1]);
    //glb file
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF\n");
    }

    return model;
}

