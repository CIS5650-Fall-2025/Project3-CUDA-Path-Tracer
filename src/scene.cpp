#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "ImGui/imgui.h"
#include "json.hpp"
#include "scene.h"
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
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            float rough = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - rough;
            newMaterial.hasRefractive = 0.0f;
        }
        else if (p["TYPE"] == "Transmitting") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            float rough = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - rough;
            newMaterial.hasRefractive = p["TRANSMITTANCE"];
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    auto makeGeomFromJson = [&](const auto& p, GeomType type, int materialId) -> Geom {
        Geom newGeom;
        newGeom.type = type;
        newGeom.materialid = materialId;
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        return newGeom;
        };


    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        int matId = MatNameToID[p["MATERIAL"]];

        if (type == "sphere")
        {
            Geom newGeom = makeGeomFromJson(p, SPHERE, matId);
            geoms.push_back(newGeom);
        }
        else if (type == "cube")
        {
            Geom newGeom = makeGeomFromJson(p, CUBE, matId);
            geoms.push_back(newGeom);
        }
        else if (type == "objMesh") {
            tinyobj::ObjReaderConfig config;
            tinyobj::ObjReader objLoader;

            std::size_t idx = jsonName.rfind("\\");
            std::string fullObjPath = jsonName.substr(0, idx + 1) + p["PATH"].get<std::string>() + ".obj";

            if (!objLoader.ParseFromFile(fullObjPath)) {
                if (!objLoader.Error().empty()) {
                    std::cerr << "OBJ Load Error: " << objLoader.Error() << std::endl;
                }
                exit(EXIT_FAILURE);
            }
            // Access geometry data
            auto& attribs = objLoader.GetAttrib();
            auto& shapes = objLoader.GetShapes();

            for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
                Geom meshGeom = makeGeomFromJson(p, CUSTOM, matId);
                auto& shapeMesh = shapes[shapeIndex].mesh;
                meshGeom.vertex_indices.x = static_cast<float>(this->vertices.size());
                glm::vec3 boundingBoxMin(FLT_MAX);
                glm::vec3 boundingBoxMax(-FLT_MAX);

                for (size_t i = 0; i < shapeMesh.num_face_vertices.size(); ++i) {
                    size_t vertsInFace = static_cast<size_t>(shapeMesh.num_face_vertices[i]);

                    for (size_t vIdx = 0; vIdx < vertsInFace; ++vIdx) {
                        Vertex vert;
                        // Retrieve the index struct for this vertex of the face
                        tinyobj::index_t idx = shapeMesh.indices[i * vertsInFace + vIdx];

                        // Position
                        vert.position = glm::vec3(
                            attribs.vertices[3 * idx.vertex_index + 0],
                            attribs.vertices[3 * idx.vertex_index + 1],
                            attribs.vertices[3 * idx.vertex_index + 2]
                        );
                        // Update bounding box
                        boundingBoxMin = glm::min(boundingBoxMin, glm::vec3(vert.position));
                        boundingBoxMax = glm::max(boundingBoxMax, glm::vec3(vert.position));

                        // Normal (if available)
                        if (idx.normal_index >= 0) {
                            vert.normal = glm::vec3(
                                attribs.normals[3 * idx.normal_index + 0],
                                attribs.normals[3 * idx.normal_index + 1],
                                attribs.normals[3 * idx.normal_index + 2]
                            );
                        }
                        else {
                            vert.normal = glm::vec3(0.0f, 0.0f, 0.0f);
                        }

                        // Texture coordinates (if available)
                        if (idx.texcoord_index >= 0) {
                            vert.uv = glm::vec2(
                                attribs.texcoords[2 * idx.texcoord_index + 0],
                                attribs.texcoords[2 * idx.texcoord_index + 1]
                            );
                        }
                        else {
                            vert.uv = glm::vec2(0.0f, 0.0f);
                        }
                        this->vertices.push_back(vert);
                    }
                }
                meshGeom.vertex_indices.y = static_cast<float>(this->vertices.size() - 1);
                meshGeom.boundingBoxMin = boundingBoxMin;
                meshGeom.boundingBoxMax = boundingBoxMax;
                geoms.push_back(meshGeom);
                }
            }
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
