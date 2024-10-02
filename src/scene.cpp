#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene()
{
    InitializeCameraAndRenderState();

}

void Scene::LoadFromFile(string filename){
    // Initialize Render State
    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = filename.substr(filename.find_last_of('.', filename.find_last_of('.') - 1) + 1,
                                      filename.find_last_of('.') - filename.find_last_of('.', filename.find_last_of('.') - 1) - 1);

    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        cout << "Successfully loaded JSON file" << endl;
        sceneReady = true;
        useBVH = false;
        useBasicBVC = false;
        return;
    }
    else if (ext == ".obj")
    {   
        string display_room_path = "../scenes/display_room.json";
        loadFromJSON(display_room_path);
        loadFromOBJ(filename);
        cout << "Successfully loaded OBJ file" << endl;
        sceneReady = true;
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::InitializeCameraAndRenderState(){
    sceneReady = false;
    // Initialize Render State
    state.iterations = 0;
    state.traceDepth = 0;
    state.imageName = "";

    // Initialize Camera with default json values
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = 800;
    camera.resolution.y = 800;
    float fovy = 45.0f;
    float eye_x = 0.0f;
    float eye_y = 5.0f;
    float eye_z = 10.5f;
    float lookat_x = 0.0f;
    float lookat_y = 5.0f;
    float lookat_z = 0.0f;
    float up_x = 0.0f;
    float up_y = 1.0f;
    float up_z = 0.0f;
    camera.position = glm::vec3(eye_x, eye_y, eye_z);
    camera.lookAt = glm::vec3(lookat_x, lookat_y, lookat_z);
    camera.up = glm::vec3(up_x, up_y, up_z);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //initialize render state image
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
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
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 0.0f;
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

void Scene::loadFromOBJ(const std::string& filename) {
    std::string warn;
    std::string err;

    tinyobj::ObjReaderConfig reader_config;
    // reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;
    int init_mat_size = materials.size();

    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!err.empty()) {
            std::cerr << "TinyObjReader: " << err << std::endl;
        }
        exit(1);
    }

    if (!warn.empty()) {
        std::cout << "TinyObjReader: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "TinyObjReader: " << err << std::endl;
    }

    mesh.attrib = reader.GetAttrib();
    mesh.shapes = reader.GetShapes();
    mesh.materials = reader.GetMaterials();

    // Populate vertices
    mesh.vertices.reserve(mesh.attrib.vertices.size() / 3);
    for (size_t i = 0; i < mesh.attrib.vertices.size(); i += 3) {
        mesh.vertices.push_back(glm::vec3(
            mesh.attrib.vertices[i],
            mesh.attrib.vertices[i + 1],
            mesh.attrib.vertices[i + 2]
        ));
    }
    if (autoCentralizeObj) {
        autoCentralize();
    }
    // Populate face indices
    for (const auto& shape : mesh.shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv == 3) {  // We're only handling triangles
                glm::ivec3 face;
                for (int v = 0; v < 3; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    face[v] = idx.vertex_index;
                }
                mesh.faceIndices.push_back(face);

                // Get the normal from the OBJ file (using the first vertex's normal)
                tinyobj::index_t idx = shape.mesh.indices[index_offset];
                if (idx.normal_index >= 0) {
                    glm::vec3 normal(
                        mesh.attrib.normals[3 * idx.normal_index + 0],
                        mesh.attrib.normals[3 * idx.normal_index + 1],
                        mesh.attrib.normals[3 * idx.normal_index + 2]
                    );
                    mesh.faceNormals.push_back(glm::normalize(normal));
                } else {
                    // If no normal is provided, calculate it
                    glm::vec3 v0 = mesh.vertices[face[0]];
                    glm::vec3 v1 = mesh.vertices[face[1]];
                    glm::vec3 v2 = mesh.vertices[face[2]];
                    glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                    mesh.faceNormals.push_back(normal);
                }
                
                // Add material for this face
                int matId = shape.mesh.material_ids[f];
                if (matId >= 0 && matId < mesh.materials.size()) {
                    const auto& objMat = mesh.materials[matId];
                    // Check if we've already added this material
                    auto it = MatNameToID.find(objMat.name);
                    if (it == MatNameToID.end()) {
                        Material mat;
                        mat.color = glm::vec3(objMat.diffuse[0], objMat.diffuse[1], objMat.diffuse[2]);
                        mat.specular.exponent = objMat.shininess;
                        mat.specular.color = glm::vec3(objMat.specular[0], objMat.specular[1], objMat.specular[2]);
                        mat.hasReflective = (objMat.illum == 3) ? 1.0f : 0.0f;  // Assuming illum 3 is reflective
                        mat.hasRefractive = (objMat.ior > 1.0f && objMat.dissolve < 1.0f) ? 1.0f : 0.0f;
                        mat.indexOfRefraction = objMat.ior;
                        mat.emittance = objMat.emission[0];  // Using the first component as emittance
                        
                        materials.push_back(mat);
                        int newIndex = materials.size() - 1;
                        MatNameToID[objMat.name] = newIndex;
                        mesh.faceMatIndices.push_back(newIndex);
                    } else {
                        // Existing material, use its index
                        mesh.faceMatIndices.push_back(it->second);
                    }
                } else {
                    // Default material (white diffuse)
                    if (MatNameToID.find("default") == MatNameToID.end()) {
                        Material mat;
                        mat.color = glm::vec3(1.0f, 1.0f, 1.0f);
                        materials.push_back(mat);
                        MatNameToID["default"] = materials.size() - 1;
                    }
                    mesh.faceMatIndices.push_back(MatNameToID["default"]);
                }
            } else {
                std::cerr << "Warning: Face with " << fv << " vertices found. Skipping." << std::endl;
            }
            index_offset += fv;
        }
    }

    std::cout << "Loaded " << mesh.vertices.size() << " vertices, " 
              << mesh.faceIndices.size() << " faces, and "
              << materials.size() - init_mat_size << " face materials." << std::endl;
    
    if (useBVH) {
        buildBVH();
    }else if (useBasicBVC) {
        max_leaf_size = mesh.faceIndices.size();
        buildBVH();
    }
}

void Scene::autoCentralize() {
    if (mesh.vertices.empty()) {
        std::cout << "No vertices or faces to centralize." << std::endl;
        return;
    }
    
    glm::vec3 geometricCenter(0.0f);
    float totalDistance = 0.0f;
    size_t numVertices = mesh.vertices.size();

    // Compute geometric center
    for (const auto& v : mesh.vertices) {
        geometricCenter += v;
    }
    geometricCenter /= static_cast<float>(numVertices);

    // Compute average distance from center
    for (const auto& v : mesh.vertices) {
        totalDistance += glm::length(v - geometricCenter);
    }
    float avgDistance = totalDistance / static_cast<float>(numVertices);

    // Use average distance as scale
    glm::vec3 scale(avgDistance, avgDistance, avgDistance);

    std::cout << "Geometric Center: " << glm::to_string(geometricCenter) << std::endl;
    std::cout << "Average Distance (Scale): " << avgDistance << std::endl;

    // Call transformToTarget with the computed bounding box information
    transformToTarget(geometricCenter, scale);
}

void Scene::transformToTarget(const glm::vec3& bboxCenter, const glm::vec3& bboxScale) {
    // Define target transformation parameters
    glm::vec3 targetTranslation(0.0f, 2.0f, 0.0f);
    float targetScale = 2.0f;
    float rotationAngle = glm::radians(45.0f);

    glm::vec3 offset = targetTranslation - bboxCenter;
    float scaleFactor = targetScale / bboxScale.x;

    // Step 1: Translation
    for (auto& vertex : mesh.vertices) {
        vertex += offset;
    }

    // Step 2: Rotation
    glm::mat4 rotationMatrix(1.0f);
    float c = cos(rotationAngle);
    float s = sin(rotationAngle);
    rotationMatrix[0][0] = c;
    rotationMatrix[0][2] = s;
    rotationMatrix[2][0] = -s;
    rotationMatrix[2][2] = c;

    for (auto& vertex : mesh.vertices) {
        vertex = glm::vec3(rotationMatrix * glm::vec4(vertex, 1.0f));
    }

    // Step 3: Scaling
    for (auto& vertex : mesh.vertices) {
        vertex = (vertex - targetTranslation) * scaleFactor + targetTranslation;
    }

    // Apply rotation to normals
    glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(rotationMatrix)));
    for (auto& normal : mesh.faceNormals) {
        normal = glm::normalize(normalMatrix * normal);
    }
}
