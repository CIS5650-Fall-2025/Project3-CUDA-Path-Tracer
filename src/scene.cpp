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

            // create the first bounding sphere generation data if it does not exist
            if (this->bounding_sphere_generations.empty()) {
                this->bounding_sphere_generations.push_back(
                    bounding_sphere_generation_data()
                );
            }

            // allocate a vector to store all the vertex data
            std::shared_ptr<std::vector<vertex_data>> pointer {
                std::make_shared<std::vector<vertex_data>>()
            };

            // obtain a reference to the allocated vector
            std::vector<vertex_data>& vertices {
                *pointer
            };

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

                    // update the material index of the vertex
                    vertex.material_index = newGeom.materialid;

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

            // transfer the loaded vertices to the first bounding sphere generation data
            this->bounding_sphere_generations.begin()->vertices.insert(
                this->bounding_sphere_generations.begin()->vertices.end(),
                vertices.begin(), vertices.end()
            );

            // skip this geometry if it holds a mesh object
            continue;
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

    // generate the bounding sphere hierarchy if the bounding sphere generation vector is not empty
    if (!this->bounding_sphere_generations.empty()) {

        // declare a vector containing all the indices of the bounding sphere generation data to process
        std::shared_ptr<std::vector<int>> working_indices {
            std::make_shared<std::vector<int>>()
        };

        // register the index of the first bounding sphere generation data
        working_indices->push_back(0);

        // keep processing the bounding sphere generation data until the number of indices is zero
        while (!working_indices->empty()) {

            // acquire the last index
            const int working_index {working_indices->back()};

            // remove the last index
            working_indices->pop_back();

            // acquire a reference to the bounding sphere generation data
            bounding_sphere_generation_data& bounding_sphere_generation {
                this->bounding_sphere_generations[working_index]
            };

            // acquire a reference to the vector of vertices in the current bounding sphere
            std::vector<vertex_data>& vertices {
                bounding_sphere_generation.vertices
            };

            // declare a variable for the minimal corner of the bounding box
            glm::vec3 minimal_corner {glm::vec3(std::numeric_limits<float>::max())};

            // declare a variable for the maximal corner of the bounding box
            glm::vec3 maximal_corner {glm::vec3(std::numeric_limits<float>::lowest())};

            // declare a variable for the average radius of the bounding spheres
            float average_radius {0.0f};

            // iterate through all the triangles
            for (std::size_t index {0}; index < vertices.size(); index += 3) {

                // compute the centroid of the triangle
                const glm::vec3 centroid {
                    (vertices[index + 0].point + vertices[index + 1].point + vertices[index + 2].point) / 3.0f
                };

                // compute the distances between the vertex positions to the centroid
                const float distance_0 {glm::distance<float>(vertices[index + 0].point, centroid)};
                const float distance_1 {glm::distance<float>(vertices[index + 1].point, centroid)};
                const float distance_2 {glm::distance<float>(vertices[index + 2].point, centroid)};

                // compute the radius of the bounding sphere
                const float radius {
                    glm::max<float>(distance_0, glm::max<float>(distance_1, distance_2))
                };

                // update the minimal corner of the bounding box
                minimal_corner.x = glm::min<float>(minimal_corner.x, centroid.x - radius);
                minimal_corner.y = glm::min<float>(minimal_corner.y, centroid.y - radius);
                minimal_corner.z = glm::min<float>(minimal_corner.z, centroid.z - radius);

                // update the maximal corner of the bounding box
                maximal_corner.x = glm::max<float>(maximal_corner.x, centroid.x + radius);
                maximal_corner.y = glm::max<float>(maximal_corner.y, centroid.y + radius);
                maximal_corner.z = glm::max<float>(maximal_corner.z, centroid.z + radius);

                // update the average radius
                average_radius += radius;
            }

            // compute the average radius
            average_radius /= static_cast<float>(vertices.size() / 3);

            // determine the center of the bounding sphere
            bounding_sphere_generation.center = (minimal_corner + maximal_corner) * 0.5f;

            // determine the radius of the bounding sphere
            bounding_sphere_generation.radius = glm::distance<float>(minimal_corner, maximal_corner) * 0.5f;

            // compute the dimension of the bounding box
            const glm::vec3 dimension {maximal_corner - minimal_corner};

            // compute the maximal length of the bounding box
            const float length {
                glm::max<float>(dimension.x, glm::max<float>(dimension.y, dimension.z)) - glm::epsilon<float>()
            };

            // determine the separation direction
            const int separation_direction {
                dimension.x >= length ? 0 : (dimension.y >= length ? 1 : 2)
            };

            // allocate two vectors to store the separated vertices
            std::shared_ptr<std::vector<vertex_data>> separated_vertices[2] {
                std::make_shared<std::vector<vertex_data>>(),
                std::make_shared<std::vector<vertex_data>>(),
            };

            // iterate through all the triangles again to perform separation
            for (std::size_t index {0}; index < vertices.size(); index += 3) {

                // compute the centroid of the triangle
                const glm::vec3 centroid {
                    (vertices[index + 0].point + vertices[index + 1].point + vertices[index + 2].point) / 3.0f
                };

                // compute the distances between the vertex positions to the centroid
                const float distance_0 {glm::distance<float>(vertices[index + 0].point, centroid)};
                const float distance_1 {glm::distance<float>(vertices[index + 1].point, centroid)};
                const float distance_2 {glm::distance<float>(vertices[index + 2].point, centroid)};

                // compute the radius of the bounding sphere
                const float radius {
                    glm::max<float>(distance_0, glm::max<float>(distance_1, distance_2))
                };

                // declare the index of the target vector to store this triangle
                int target_index;

                // perform separation along the x direction
                if (separation_direction == 0) {
                    target_index = centroid.x + radius - average_radius < bounding_sphere_generation.center.x ? 0 : 1;

                    // perform separation along the y direction
                } else if (separation_direction == 1) {
                    target_index = centroid.y + radius - average_radius < bounding_sphere_generation.center.y ? 0 : 1;

                    // perform separation along the z direction
                } else {
                    target_index = centroid.z + radius - average_radius < bounding_sphere_generation.center.z ? 0 : 1;
                }

                // store the vertices in the target vector
                separated_vertices[target_index]->push_back(vertices[index + 0]);
                separated_vertices[target_index]->push_back(vertices[index + 1]);
                separated_vertices[target_index]->push_back(vertices[index + 2]);
            }

            // proceed to the next iteration with either vector containing the separated vertices is empty
            if (separated_vertices[0]->empty() || separated_vertices[1]->empty()) {
                continue;
            }

            // update the child indices of the current bounding sphere generation data
            for (std::size_t index {0}; index < 2; index += 1) {
                bounding_sphere_generation.child_indices[index] = static_cast<int>(
                    this->bounding_sphere_generations.size() + index
                );
            }

            // remove all the vertices in the current bounding sphere generation data as they will be transfered
            bounding_sphere_generation.vertices.clear();

            // create two new bounding sphere generation data as children
            for (std::size_t index {0}; index < 2; index += 1) {
                this->bounding_sphere_generations.push_back(
                    bounding_sphere_generation_data()
                );

                // transfer the vertices to the new bounding sphere generation data
                this->bounding_sphere_generations.back().vertices = *separated_vertices[index];

                // store the index of the new bounding sphere generation data
                working_indices->push_back(static_cast<int>(
                    this->bounding_sphere_generations.size() - 1
                ));
            }
        }

        // print out a message
        std::cout << "Generated " << this->bounding_sphere_generations.size() << " bounding spheres" << std::endl;

        // generate the bounding sphere data from the bounding sphere generation data
        for (const bounding_sphere_generation_data& bounding_sphere_generation : this->bounding_sphere_generations) {

            // create a new bounding sphere
            this->bounding_spheres.push_back(
                bounding_sphere_data()
            );

            // obtain a reference to the new bounding sphere
            bounding_sphere_data& bounding_sphere {
                this->bounding_spheres.back()
            };

            // update the center of the new bounding sphere
            bounding_sphere.center = bounding_sphere_generation.center;

            // update the radius of the new bounding sphere
            bounding_sphere.radius = bounding_sphere_generation.radius;

            // update the child indices of the new bounding sphere
            bounding_sphere.child_indices[0] = bounding_sphere_generation.child_indices[0];
            bounding_sphere.child_indices[1] = bounding_sphere_generation.child_indices[1];

            // update the vertex data of the new bounding sphere when the bounding sphere generation data contain vertices
            if (!bounding_sphere_generation.vertices.empty()) {

                // update the index of the first vertex for the new bounding sphere
                bounding_sphere.index = static_cast<int>(this->vertices.size());

                // update the number of triangles for the new bounding sphere
                bounding_sphere.count = static_cast<int>(bounding_sphere_generation.vertices.size() / 3);

                // transfer all the vertices to the vector containing all the vertices in the scene
                this->vertices.insert(
                    this->vertices.end(),
                    bounding_sphere_generation.vertices.begin(),
                    bounding_sphere_generation.vertices.end()
                );
            }
        }

        // remove all the bounding sphere generation data
        this->bounding_sphere_generations.clear();

        // print out a message
        std::cout << "Generated " << this->vertices.size() << " vertices" << std::endl;
    }
}
