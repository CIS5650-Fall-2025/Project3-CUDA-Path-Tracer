#include "meshLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tinyobjloader/tiny_obj_loader.h"

#define TINYGLTF_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"

bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.rfind(suffix) == (str.size() - suffix.size());
}

/**
 * @brief This function loads an OBJ file and stores the vertices, normals, and UVs in the faces vector.
 * The code is taken from the tinyobjloader repository: https://github.com/tinyobjloader/tinyobjloader.
 * Although the original implementation can read meshes with arbitrarily-shaped faces, we are assuming that
 * the faces are triangles.
 * 
 * @param filepath The absolute path to the OBJ file.
 */
void loadOBJ(const std::string &filepath, std::vector<Triangle> &faces) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config)) {
        if (!reader.Error().empty()) {
            printf("TinyObjReader ERROR: %s\n", reader.Error().c_str());
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        printf("TinyObjReader WARNING: %s\n", reader.Warning().c_str());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            
            if (fv != 3) {
                std::cerr << "This OBJ loader only supports triangles. Exiting..." << std::endl;
                exit(1);
            }

            std::vector<glm::vec3> verticesForOneFace;
            std::vector<glm::vec3> normalsForOneFace;
            std::vector<glm::vec2> uvsForOneFace;
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
                verticesForOneFace.push_back(glm::vec3(vx, vy, vz));

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                    tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                    tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                    normalsForOneFace.push_back(glm::vec3(nx, ny, nz));
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                    tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
                    uvsForOneFace.push_back(glm::vec2(tx, ty));
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            
            Triangle t(verticesForOneFace[0], verticesForOneFace[1], verticesForOneFace[2]);
            if (normalsForOneFace.size() > 0) {
                for (int i = 0; i < fv; i++) {
                    t.normals[i] = normalsForOneFace[i];
                }
            }
            if (uvsForOneFace.size() > 0) {
                for (int i = 0; i < fv; i++) {
                    t.uvs[i] = uvsForOneFace[i];
                }
            }
            faces.push_back(t);

            // per-face material
            shapes[s].mesh.material_ids[f];
            index_offset += fv;
        }
    }
}

// Function to compute the transformation matrix for a node
glm::mat4 getNodeTransform(const tinygltf::Node &node) {
    glm::mat4 transform = glm::mat4(1.0f); // Identity matrix

    // Apply translation, if present
    if (!node.translation.empty()) {
        glm::vec3 translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        transform = glm::translate(transform, translation);
    }

    // Apply rotation, if present
    if (!node.rotation.empty()) {
        glm::quat rotation = glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]); // Quaternion
        transform *= glm::mat4_cast(rotation); // Convert quaternion to matrix and multiply
    }

    // Apply scale, if present
    if (!node.scale.empty()) {
        glm::vec3 scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        transform = glm::scale(transform, scale);
    }

    return transform;
}

// Helper function to extract and populate triangle data
void populateTriangles(const int mode, std::vector<Triangle> &faces, const tinygltf::Accessor &indexAccessor, const unsigned short *indices, 
                       const float *positions, const float *normals, const float *uvs, const glm::mat4 &transform) {
    
    unsigned short idx0;
    unsigned short idx1;
    unsigned short idx2;
    
    // Iterate over the indices in sets of 3 (triangles)
    for (size_t i = 0; i < indexAccessor.count; i += 3) {
        if (mode == TINYGLTF_MODE_TRIANGLES) {
            idx0 = indices[i + 0];
            idx1 = indices[i + 1];
            idx2 = indices[i + 2];
        }
        else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
            idx0 = indices[i - 2];
            idx1 = indices[i - 1];
            idx2 = indices[i];
                
            if (i % 2 != 0) std::swap(idx1, idx2);  // Flip the winding order for odd-numbered triangles
        }
        else if (mode == TINYGLTF_MODE_TRIANGLE_FAN) {
            idx0 = indices[0];
            idx1 = indices[i - 1];
            idx2 = indices[i];
        }
        
        // Get vertex positions and apply transformation
        glm::vec3 p1 = glm::vec3(positions[idx0 * 3 + 0], positions[idx0 * 3 + 1], positions[idx0 * 3 + 2]);
        glm::vec3 p2 = glm::vec3(positions[idx1 * 3 + 0], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
        glm::vec3 p3 = glm::vec3(positions[idx2 * 3 + 0], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);

        // Apply transformation
        p1 = glm::vec3(transform * glm::vec4(p1, 1.0f));
        p2 = glm::vec3(transform * glm::vec4(p2, 1.0f));
        p3 = glm::vec3(transform * glm::vec4(p3, 1.0f));

        // Get vertex normals (without applying transformation)
        glm::vec3 n1 = normals ? glm::vec3(normals[idx0 * 3 + 0], normals[idx0 * 3 + 1], normals[idx0 * 3 + 2])
                               : glm::normalize(glm::cross(p2 - p1, p3 - p2));
        glm::vec3 n2 = normals ? glm::vec3(normals[idx1 * 3 + 0], normals[idx1 * 3 + 1], normals[idx1 * 3 + 2]) : n1;
        glm::vec3 n3 = normals ? glm::vec3(normals[idx2 * 3 + 0], normals[idx2 * 3 + 1], normals[idx2 * 3 + 2]) : n1;

        // Get vertex UVs
        glm::vec2 uv1 = uvs ? glm::vec2(uvs[idx0 * 2 + 0], uvs[idx0 * 2 + 1]) : glm::vec2(0.0f);
        glm::vec2 uv2 = uvs ? glm::vec2(uvs[idx1 * 2 + 0], uvs[idx1 * 2 + 1]) : glm::vec2(0.0f);
        glm::vec2 uv3 = uvs ? glm::vec2(uvs[idx2 * 2 + 0], uvs[idx2 * 2 + 1]) : glm::vec2(0.0f);

        // Create Triangle and populate faces vector
        Triangle tri;
        glm::vec3 points[3] = {p1, p2, p3};
        glm::vec3 normalsArr[3] = {n1, n2, n3};
        glm::vec2 uvsArr[3] = {uv1, uv2, uv3};

        for (int i = 0; i < 3; ++i) {
            tri.points[i] = points[i];
            tri.normals[i] = normalsArr[i];
            tri.uvs[i] = uvsArr[i];
        }

        tri.planeNormal = glm::normalize(glm::cross(p2 - p1, p3 - p2));

        faces.push_back(tri);
    }
}

// Recursive function to traverse nodes and extract mesh data
void extractMeshDataFromGLTF(const tinygltf::Model &model, int nodeIndex, std::vector<Triangle> &faces, const glm::mat4 &parentTransform) {
    const tinygltf::Node &node = model.nodes[nodeIndex];
    
    // Compute the transformation matrix for this node
    glm::mat4 nodeTransform = parentTransform * getNodeTransform(node);

    // If the node contains a mesh, process it
    if (node.mesh >= 0) {
        const tinygltf::Mesh &mesh = model.meshes[node.mesh];
        
        for (const auto &primitive : mesh.primitives) {
            // Get POSITION attribute (vertex positions)
            const auto posIt = primitive.attributes.find("POSITION");
            if (posIt == primitive.attributes.end()) {
                std::cerr << "No POSITION attribute found" << std::endl;
                continue;
            }
            const tinygltf::Accessor &posAccessor = model.accessors[posIt->second];
            const tinygltf::BufferView &posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer &posBuffer = model.buffers[posBufferView.buffer];
            const float *positions = reinterpret_cast<const float *>(
                &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

            // Get NORMAL attribute (vertex normals), if available
            const float *normals = nullptr;
            const auto normIt = primitive.attributes.find("NORMAL");
            if (normIt != primitive.attributes.end()) {
                const tinygltf::Accessor &normAccessor = model.accessors[normIt->second];
                const tinygltf::BufferView &normBufferView = model.bufferViews[normAccessor.bufferView];
                const tinygltf::Buffer &normBuffer = model.buffers[normBufferView.buffer];
                normals = reinterpret_cast<const float *>(
                    &normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);
            }

            // Get TEXCOORD_0 attribute (UVs), if available
            const float *uvs = nullptr;
            const auto uvIt = primitive.attributes.find("TEXCOORD_0");
            if (uvIt != primitive.attributes.end()) {
                const tinygltf::Accessor &uvAccessor = model.accessors[uvIt->second];
                const tinygltf::BufferView &uvBufferView = model.bufferViews[uvAccessor.bufferView];
                const tinygltf::Buffer &uvBuffer = model.buffers[uvBufferView.buffer];
                uvs = reinterpret_cast<const float *>(
                    &uvBuffer.data[uvBufferView.byteOffset + uvAccessor.byteOffset]);
            }

            // Access indices if they exist (assuming unsigned short type)
            if (primitive.indices >= 0) {
                const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer &indexBuffer = model.buffers[indexBufferView.buffer];
                const unsigned short *indices = reinterpret_cast<const unsigned short *>(
                    &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                
                if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != TINYGLTF_MODE_TRIANGLE_STRIP && primitive.mode != TINYGLTF_MODE_TRIANGLE_FAN) {
                    std::cerr << "Unsupported primitive mode: " << primitive.mode << std::endl;
                    continue;
                }
                
                populateTriangles(primitive.mode, faces, indexAccessor, indices, positions, normals, uvs, nodeTransform);  
            }
        }
    }

    // Recursively traverse child nodes
    for (size_t i = 0; i < node.children.size(); ++i) {
        extractMeshDataFromGLTF(model, node.children[i], faces, nodeTransform);
    }
}

// Entry function to traverse the GLTF scene
void extractMeshDataFromGLTFScene(const tinygltf::Model &model, std::vector<Triangle> &faces) {
    const glm::mat4 identityMatrix = glm::mat4(1.0f); // Identity matrix

    // Start with the root nodes in the default scene
    const tinygltf::Scene &scene = model.scenes[model.defaultScene];
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        extractMeshDataFromGLTF(model, scene.nodes[i], faces, identityMatrix);
    }
}

void loadGLTFOrGLB(const std::string &filepath, std::vector<Triangle> &faces) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret;
    if (endsWith(filepath, ".gltf")) {
        printf("Loading GLTF file: %s\n", filepath.c_str());
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
    }
    else if (endsWith(filepath, ".glb")) {
        printf("Loading GLB file: %s\n", filepath.c_str());
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
    }

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF\n");
        exit(-1);
    }

    extractMeshDataFromGLTFScene(model, faces);
}