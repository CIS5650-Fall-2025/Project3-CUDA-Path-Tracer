#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    } else if (ext == ".gltf" || ext == ".glb")
    {
        load_from_gltf(filename);
    } else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

Scene::~Scene() {
    for (auto &mesh : this->meshes) {
        free(mesh.vertices);
        free(mesh.indices);
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
            newMaterial.specular.color = newMaterial.color;
            const float roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1 - roughness;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
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

    if (cameraData.contains("FOCAL_DISTANCE")) {
        const auto &focal_distance = cameraData["FOCAL_DISTANCE"];
        camera.focal_distance = focal_distance;
    }

    if (cameraData.contains("LENS_RADIUS")) {
        const auto &lens_radius = cameraData["LENS_RADIUS"];
        camera.lens_radius = lens_radius;
    }

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

void Scene::parse_gltf_node(const tinygltf::Model &model, int node_id, const glm::mat4 &base_transform) {
    const auto &node = model.nodes[node_id];    

    glm::mat4 transform;
    glm::vec3 translation;
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};

    if (node.matrix.size() == 16) {
        transform = glm::make_mat4(node.matrix.data());
    } else {
        if (node.translation.size() == 3) translation = glm::make_vec3(node.translation.data());
        if (node.rotation.size() == 3) {
            rotation = glm::quat(glm::make_vec3(node.rotation.data()));
        } else if (node.rotation.size() == 4) {
            rotation = glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
        }
        if (node.scale.size() == 3) scale = glm::make_vec3(node.scale.data());

        transform = glm::translate(glm::mat4(), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
    }

    transform = base_transform * transform;
    glm::mat4 inverse_transform = glm::inverse(transform);
    glm::mat4 inverse_transpose = glm::inverseTranspose(transform);

    if (node.mesh != -1) { // mesh

        for (const auto &primitive : model.meshes[node.mesh].primitives) {
            Mesh mesh;
            mesh.translation = translation;
            mesh.rotation = rotation;
            mesh.scale = scale;
            mesh.transform = transform;
            mesh.inverseTransform = inverse_transform;
            mesh.invTranspose = inverse_transpose;

            if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
                const auto &accessor = model.accessors[primitive.attributes.find("POSITION")->second];
                const auto &bufferView = model.bufferViews[accessor.bufferView];
                const auto &buffer = model.buffers[bufferView.buffer];
                const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

                mesh.num_vertices = accessor.count;
                mesh.vertices = (glm::vec3 *) malloc(accessor.count * sizeof(glm::vec3));
                for (auto i = 0; i < accessor.count; ++i) {
                    mesh.vertices[i] = glm::vec3{positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]};
                }
            }

            // Parse indices
            if (primitive.indices > -1) {
                const auto &indexAccessor = model.accessors[primitive.indices];
                const auto &bufferView = model.bufferViews[indexAccessor.bufferView];
                const auto &buffer = model.buffers[bufferView.buffer];

                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const unsigned int* indices = reinterpret_cast<const unsigned int*>(&buffer.data[bufferView.byteOffset + indexAccessor.byteOffset]);

                    mesh.num_indices = indexAccessor.count;
                    mesh.indices = (int *) malloc(indexAccessor.count * sizeof(int));
                    for (auto i = 0; i < indexAccessor.count; ++i) {
                        mesh.indices[i] = indices[i];
                    }

                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const unsigned short* indices = reinterpret_cast<const unsigned short*>(&buffer.data[bufferView.byteOffset + indexAccessor.byteOffset]);
                    mesh.num_indices = indexAccessor.count;
                    mesh.indices = (int *) malloc(indexAccessor.count * sizeof(int));
                    for (auto i = 0; i < indexAccessor.count; ++i) {
                        mesh.indices[i] = indices[i];
                    }
                }
            }

            Material new_material;
            if (primitive.material != -1) {
                const auto &material = model.materials[primitive.material];
                new_material.color = glm::vec3{material.pbrMetallicRoughness.baseColorFactor[0],
                                                material.pbrMetallicRoughness.baseColorFactor[1],
                                                material.pbrMetallicRoughness.baseColorFactor[2]};

                new_material.hasReflective = material.pbrMetallicRoughness.metallicFactor;
                new_material.specular.exponent = 1.0f;
                new_material.specular.color = new_material.color;

                new_material.emittance = glm::length(glm::vec3{material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]});

                new_material.hasRefractive = 0.0f;

            }
            materials.push_back(new_material);
            mesh.material_id = materials.size() - 1;

            mesh.compute_bounding_box();
            meshes.push_back(mesh);
        }
    } else if (node.camera != -1) {
        const auto &camera = model.cameras[node.camera];
        if (camera.type == "perspective") {
            auto &new_camera = state.camera;

            new_camera.resolution.x = 800;
            new_camera.resolution.y = 800;

            const auto y_fov = camera.perspective.yfov * (180 / PI);
            new_camera.fov = glm::vec2(y_fov * camera.perspective.aspectRatio, y_fov);

            new_camera.position = translation;
            new_camera.up = glm::vec3{0.0f, 1.0f, 0.0f};
            new_camera.right = glm::normalize(glm::cross(new_camera.view, new_camera.up));

            const auto yscaled = tan(y_fov * (PI / 180));
            const auto xscaled = (yscaled * new_camera.resolution.x) / new_camera.resolution.y;
            new_camera.pixelLength = glm::vec2{
                2 * xscaled / (float)new_camera.resolution.x,
                2 * yscaled / (float)new_camera.resolution.y
            };

            new_camera.view = glm::normalize(new_camera.lookAt - new_camera.position);

            const auto array_len = new_camera.resolution.x * new_camera.resolution.y;
            state.image.resize(array_len);
            std::fill(state.image.begin(), state.image.end(), glm::vec3{});
            state.iterations = 2000;
            state.traceDepth = 8;
        }
    } else if (node.name == "Camera") {
        auto &new_camera = state.camera;
        
        new_camera.resolution.x = 800;
        new_camera.resolution.y = 800;

        const auto y_fov = 60.0f;
        new_camera.fov = glm::vec2(y_fov, y_fov);

        new_camera.position = glm::vec3(transform[3][0], transform[3][1], transform[3][2]);
        new_camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
        new_camera.right = glm::normalize(glm::cross(new_camera.view, new_camera.up));

        const auto yscaled = tan(y_fov * (PI / 180));
        const auto xscaled = (yscaled * new_camera.resolution.x) / new_camera.resolution.y;
        new_camera.pixelLength = glm::vec2{
            2 * xscaled / (float)new_camera.resolution.x,
            2 * yscaled / (float)new_camera.resolution.y
        };

        new_camera.view = glm::normalize(new_camera.lookAt - new_camera.position);

        const auto array_len = new_camera.resolution.x * new_camera.resolution.y;
        state.image.resize(array_len);
        std::fill(state.image.begin(), state.image.end(), glm::vec3{});
        state.iterations = 2000;
        state.traceDepth = 8;
    }

    for (int child : node.children) {
        parse_gltf_node(model, child, transform);
    }
}

void Scene::load_from_gltf(const std::string &gltf_filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    // Load GLTF file (either ASCII .gltf or binary .glb)
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltf_filename);
    if (!warn.empty()) std::cout << "Warning: " << warn << std::endl;
    if (!ret || !err.empty())
    {
        std::cerr << "Error: " << err << std::endl;
        exit(EXIT_FAILURE);
    }

    for (const auto &scene : model.scenes) {
        for (int node_id : scene.nodes) {
            parse_gltf_node(model, node_id, glm::mat4{});
        }
    }

    Material lightMaterial1;
    lightMaterial1.color = glm::vec3(1.0f, 0.5f, 0.5f);
    lightMaterial1.emittance = 150.0f;
    materials.push_back(lightMaterial1);

    Material lightMaterial2;
    lightMaterial2.color = glm::vec3(0.5f, 0.5f, 1.0f);
    lightMaterial2.emittance = 150.0f;
    materials.push_back(lightMaterial2);

    float x_spacing = 350.0f;
    float z_spacing = 350.0f;
    int num_x = 3;
    int num_z = 3;
    float x_start = -num_x * x_spacing / 2.0f;
    float z_start = -num_z * z_spacing / 2.0f;

    for (int i=0; i<num_x; i++) {
        for (int j=0; j<num_z; j++) {
            for (int k=-1; k<=1; k+=2) {
                Geom light;
                light.type = SPHERE;
                if (k == -1) {
                    light.materialid = materials.size()-2;
                }
                else {
                    light.materialid = materials.size()-1;
                }
                light.translation = glm::vec3(i*x_spacing+x_start, k * 400.0f, j * z_spacing + z_start);
                light.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
                light.scale = glm::vec3(70.f, 70.f, 70.f);
                light.transform = utilityCore::buildTransformationMatrix(
                    light.translation, light.rotation, light.scale);
                light.inverseTransform = glm::inverse(light.transform);
                light.invTranspose = glm::inverseTranspose(light.transform);
                geoms.push_back(light);
            }
        }
    }
    
    Material groundMaterial;
    groundMaterial.color = glm::vec3(0.2f, 0.4f, 0.2f);
    groundMaterial.emittance = 0.0f;
    groundMaterial.hasReflective = 0.0f;
    materials.push_back(groundMaterial);

    Geom ground;
    ground.type = CUBE;
    ground.materialid = materials.size()-1;
    ground.translation = glm::vec3(0.0f, -300.0f, 0.0f);
    ground.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    ground.scale = glm::vec3(100000.0f, 0.001f, 100000.0f);
    ground.transform = utilityCore::buildTransformationMatrix(
        ground.translation, ground.rotation, ground.scale);
    ground.inverseTransform = glm::inverse(ground.transform);
    ground.invTranspose = glm::inverseTranspose(ground.transform);
    geoms.push_back(ground);
}

