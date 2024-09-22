#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"

// include the tiny obj loader header without implementation
#include "tiny_obj_loader.h"

using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            // acquire the roughness of the material
            const float& roughness = p["ROUGHNESS"];

            // update the material's reflectivity
            newMaterial.hasReflective = 1.0f - roughness;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        
        // set the new geometry's type based on input data
        newGeom.type = type == "mesh" ? MESH : (type == "cube" ? CUBE : SPHERE);

        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // load the mesh data if the geometry type is MESH
        if (newGeom.type == MESH) {

            // declare the path of the obj file
            std::string path {jsonName};

            // find the position of the last slash character
            const std::size_t position {path.find_last_of('/')};

            // remove all characters after the last slash character if the position is valid
            if (position != std::string::npos) {
                path.erase(position);
            }

            // append the file name of the obj file
            path += std::string("/meshes/") + std::string(p["NAME"]) + std::string(".obj");

            // declare the obj file reader
            tinyobj::ObjReader reader {};

            // declare the reader configuration
            const tinyobj::ObjReaderConfig configuration {};

            // read the obj file
            if (!reader.ParseFromFile(path, configuration)) {

                // print the error in case of failure
                if (!reader.Error().empty()) {
                    std::cerr << "Fatal Error: " << reader.Error() << std::endl;
                }

                // make the program crash in case of failure
                std::abort();
            }

            // declare a vector to store all the vertex data
            std::vector<vertex_data> vertices {};

            // declare the offset for the indices
            std::size_t offset {0};

            // acquire the attributes
            const tinyobj::attrib_t& attributes {reader.GetAttrib()};

            // acquire the first shape
            const tinyobj::shape_t& shape {*reader.GetShapes().begin()};

            // acquire the number of faces in the shape
            const std::size_t face_count {shape.mesh.num_face_vertices.size()};

            // iterate through all the faces
            for (std::size_t face_index {0}; face_index < face_count; face_index += 1) {

                // acquire the number of vertices in the face
                const std::size_t vertex_count {static_cast<std::size_t>(shape.mesh.num_face_vertices[face_index])};

                // make sure that the face is valid and triangulated
                if (vertex_count != 3) {
                    std::cerr << "Fatal Error: encountered an invalid or untriangulated face!" << std::endl;
                }

                // iterate through all the vertices in the current face
                for (std::size_t index {0}; index < vertex_count; index += 1) {

                    // acquire the current vertex index
                    const tinyobj::index_t& vertex_index {shape.mesh.indices[offset + index]};

                    // create a new vertex
                    vertex_data vertex {};

                    // update the location of the vertex
                    vertex.point = glm::vec3(
                        attributes.vertices[static_cast<std::size_t>(vertex_index.vertex_index) * 3 + 0],
                        attributes.vertices[static_cast<std::size_t>(vertex_index.vertex_index) * 3 + 1],
                        attributes.vertices[static_cast<std::size_t>(vertex_index.vertex_index) * 3 + 2]
                    );

                    // update the normal of the vertex if it exists
                    if (vertex_index.normal_index >= 0) {
                        vertex.normal = glm::vec3(
                            attributes.normals[static_cast<std::size_t>(vertex_index.normal_index) * 3 + 0],
                            attributes.normals[static_cast<std::size_t>(vertex_index.normal_index) * 3 + 1],
                            attributes.normals[static_cast<std::size_t>(vertex_index.normal_index) * 3 + 2]
                        );
                    }

                    // update the texture coordinate of the vertex if it exists
                    if (vertex_index.texcoord_index >= 0) {
                        vertex.coordinate = glm::vec2(
                            attributes.texcoords[static_cast<std::size_t>(vertex_index.texcoord_index) * 2 + 0],
                            attributes.texcoords[static_cast<std::size_t>(vertex_index.texcoord_index) * 2 + 1]
                        );
                    }

                    // store the new vertex
                    vertices.push_back(vertex);
                }

                // increase the offset
                offset += vertex_count;
            }

            // iterate through all the faces again to compute the vertex tangent
            for (std::size_t index {0}; index < vertices.size(); index += 3) {

                // compute the vertex indices
                const std::size_t index_0 {index + 0};
                const std::size_t index_1 {index + 1};
                const std::size_t index_2 {index + 2};

                // compute the offset vectors
                const glm::vec3 vector_0 {vertices[index_1].point - vertices[index_0].point};
                const glm::vec3 vector_1 {vertices[index_2].point - vertices[index_0].point};
                const glm::vec2 vector_2 {vertices[index_1].coordinate - vertices[index_0].coordinate};
                const glm::vec2 vector_3 {vertices[index_2].coordinate - vertices[index_0].coordinate};

                // compute the denominator in tangent computation
                const float denominator {vector_2.x * vector_3.y - vector_2.y * vector_3.x};

                // proceed to the next iteration when the denominator is zero
                if (denominator == 0.0f) {
                    continue;
                }

                // compute the raw tangent
                const glm::vec3 tangent {(vector_0 * vector_3.y - vector_2.y * vector_1) / denominator};

                // iterate through all the vertices in the face to compute the actual vertex tangent
                for (std::size_t vertex_index {index}; vertex_index < index + 3; vertex_index += 1) {

                    // store the raw tangent as the vertex tangent
                    vertices[vertex_index].tangent = tangent;

                    // correct the vertex tangent based on vertex normal
                    vertices[vertex_index].tangent -= vertices[vertex_index].normal * glm::dot(
                        vertices[vertex_index].tangent, vertices[vertex_index].normal
                    );

                    // normalize the vertex tangent
                    vertices[vertex_index].tangent = glm::normalize(
                        vertices[vertex_index].tangent
                    );
                }
            }

            // print out a message
            std::cout << "Loaded " << vertices.size() << " vertices from " << path << std::endl;
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
