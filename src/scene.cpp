#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include <stb_image.h>
#include <cuda_runtime.h>


#define TINYOBJLOADER_IMPLEMENTATION
#include "./thirdparty/tinyobj_loader/tiny_obj_loader.h"

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

        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emissive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Metallic")
        {
            const auto& col = p["RGB"];
            newMaterial.hasReflective = 1.0f;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.indexOfRefraction = p.contains("IOR") ? float(p["IOR"]) : 1.0f;

            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Dielectric")
        {
            const auto& col = p["RGB"];
            newMaterial.hasRefractive = 1.0f;

            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.specular.color = glm::vec3(1.0f);
        }
        else if (p["TYPE"] == "Environment") {
            const std::string envPath = p["HDR_MAP"];
            if (p.contains("INTENSITY")) {
                newMaterial.envMap_intensity = float(p["INTENSITY"]);
            }
            
            newMaterial.is_env = true;
            if (!loadTexture(envPath, newMaterial.envMapData)) {
                std::cerr << "Failed to load env Path for Lighting " << name << "\n";
            }
        }

        if (p.contains("ALBEDO_MAP")) {
            const std::string albedoPath = p["ALBEDO_MAP"];
            if (!loadTexture(albedoPath, newMaterial.albedoMapData)) {
                std::cerr << "Failed to load albedo map for material " << name << "\n";
            }
        }

        if (p.contains("NORMAL_MAP")) {
            const std::string normalPath = p["NORMAL_MAP"];
            if (!loadTexture(normalPath, newMaterial.normalMapData)) {
                std::cerr << "Failed to load normal map for material " << name << "\n";
            }
        }


        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.materialid = MatNameToID[p["MATERIAL"]];

        // Handle transformationss
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

        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;

            const std::string filepath = p["FILEPATH"];
            LoadFromOBJ(filepath, newGeom);
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



bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void Scene::LoadFromOBJ(const std::string& filepath, Geom& geom)
{
    if (!fileExists(filepath)) {
        std::cerr << "File not found: " << filepath << std::endl;
    }


    // TinyOBJ structures to hold loaded geometry data
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filepath.c_str());

    if (!warn.empty()) std::cout << "TinyOBJ warning: " << warn << "\n";
    if (!err.empty()) std::cerr << "TinyOBJ error: " << err << "\n";

    if (!ret) {
        std::cerr << "Failed to load OBJ file: " << filepath << "\n";
        return;
    }

    std::vector<Triangle> tris;

    // Iterate over all shapes in the OBJ file
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        const auto& mesh = shape.mesh;

        // Loop over all faces
        for (size_t f = 0; f < mesh.num_face_vertices.size(); f++) {
            size_t fv = mesh.num_face_vertices[f];

            // Only support triangular faces
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            glm::vec3 v[3];  // Vertex positions
            glm::vec2 uv[3]; // Texture coordinates
            bool valid = true;

            // Extract each of the three vertices
            for (size_t i = 0; i < 3; ++i) {
                tinyobj::index_t idx = mesh.indices[index_offset + i];

                // Skip invalid indices
                if (idx.vertex_index < 0) {
                    valid = false;
                    break;
                }

                // Load vertex position
                v[i] = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                // Load texture coordinates if available
                if (!attrib.texcoords.empty() && idx.texcoord_index >= 0) {
                    uv[i] = glm::vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1] // Flip Y
                    );
                }
                else {
                    uv[i] = glm::vec2(0.0f);
                }
            }

            if (!valid) {
                index_offset += fv;
                continue;
            }

            // Transform vertices from object to world space
            glm::vec4 v0t = geom.transform * glm::vec4(v[0], 1.0f);
            glm::vec4 v1t = geom.transform * glm::vec4(v[1], 1.0f);
            glm::vec4 v2t = geom.transform * glm::vec4(v[2], 1.0f);

            // Create triangle and assign transformed vertices and UVs
            Triangle tri;
            tri.v0 = glm::vec3(v0t) / v0t.w;
            tri.v1 = glm::vec3(v1t) / v1t.w;
            tri.v2 = glm::vec3(v2t) / v2t.w;

            tri.uv0 = uv[0];
            tri.uv1 = uv[1];
            tri.uv2 = uv[2];

            // Compute flat normal for triangle (not vertex normals)
            tri.normal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));

            tris.push_back(tri);
            index_offset += fv;
        }
    }

    std::cout << "Parsed " << tris.size() << " triangles from OBJ\n";

    // NOTE: constructBVH_SAH_Binned reorders tris in-place during BVH construction
    // So we must build the BVH *before* copying the triangles into geom
    std::vector<BVHNode> bvhNodes;
    constructBVH_MidpointSplit(bvhNodes, tris, 0, static_cast<int>(tris.size()));
    //constructBVH_SAH_Binned(bvhNodes, tris, 0, static_cast<int>(tris.size()));

    // Allocate and copy triangles into the geom structure
    geom.num_triangles = static_cast<int>(tris.size());
    geom.triangles = new Triangle[geom.num_triangles];
    std::copy(tris.begin(), tris.end(), geom.triangles);

    // Allocate and copy the BVH nodes
    geom.num_BVHNodes = static_cast<int>(bvhNodes.size());
    geom.bvhNodes = new BVHNode[geom.num_BVHNodes];
    std::copy(bvhNodes.begin(), bvhNodes.end(), geom.bvhNodes);
}


