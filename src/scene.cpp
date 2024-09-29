#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "stb_image.h"

static float to_linear(float f) {
    if (f > 0.04045f) {
        return std::pow((f + 0.055f) / 1.055f, 2.4f);
    } else {
        return f / 12.92f;
    }
}

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

void Scene::loadImage(const std::string& filepathImage, int& index_, int& width_, int& height_) {
    // Set first pixel to bottom left:
    stbi_set_flip_vertically_on_load(true);

    int width, height, channels;
    uint8_t* data = stbi_load(filepathImage.c_str(), &width, &height, &channels, 0);

    if (!data) throw std::runtime_error("Failed to load image from " + filepathImage + ": " + std::string(stbi_failure_reason()));
    if (channels < 3) throw std::runtime_error("Image loaded from " + filepathImage + " has fewer than 3 color channels.");

    index_ = textures.size();
    width_ = width;
    height_ = height;

    for (int i = 0; i < width * height * channels; i += channels) {
        float r = data[i] / 255.0f;
        float g = data[i + 1] / 255.0f;
        float b = data[i + 2] / 255.0f;
        float a = 1.0f;
        if (channels == 4) a = data[i + 3] / 255.0f;

        // Convert loaded image from sRGB to linear rgb colorspace
        textures.push_back(glm::vec4(to_linear(r), to_linear(g), to_linear(b), a));
    }

    stbi_image_free(data);

    printf("Image texture loaded at index %d \n", index_);
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;

    int index, width, height;

    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Lambertian" || p["TYPE"] == "Diffuse")
        {
            newMaterial.type = LAMBERTIAN;
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.type = EMISSIVE;
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Metal" || p["TYPE"] == "Specular")
        {
            newMaterial.type = METAL;
            const float& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == "Dielectric" || p["TYPE"] == "Glass")
        {
            newMaterial.type = DIELECTRIC;
            const float& ior = p["IOR"];
            newMaterial.indexOfRefraction = ior;
        }
        else
        {
            // Default into black lambertian
            newMaterial.type = LAMBERTIAN;
            newMaterial.color = glm::vec3(0.0f);
            MatNameToID[name] = materials.size();
            materials.emplace_back(newMaterial);
            continue;
        }

        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        const auto& texType = p.value("TEXTYPE", 0);
        newMaterial.texType = CONSTANT;

        if (texType == CHECKER) {
            newMaterial.texType = CHECKER;
            const auto& checkerScale = p.value("CHECKERSCALE", 1.f);
            newMaterial.checkerScale = checkerScale;
        } else if (texType == IMAGE) {
            newMaterial.texType = IMAGE;

            // Inspired by Scotty3D's texture loading
            // https://github.com/CMU-Graphics/Scotty3D
            const std::string& filepathImage = p["FILEPATH_IMAGE"];
            loadImage(filepathImage, index, width, height);

            newMaterial.imageTextureInfo.index = index;
            newMaterial.imageTextureInfo.width = width;
            newMaterial.imageTextureInfo.height = height;
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
            newGeom.numVertices = 2;
            newGeom.vertices[0] = glm::vec3(-1.f);
            newGeom.vertices[1] = glm::vec3(1.f);
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            newGeom.numVertices = 2;
            newGeom.vertices[0] = glm::vec3(-1.f);
            newGeom.vertices[1] = glm::vec3(1.f);
        }
        else if (type == "mesh")
        {
            // Based on tinyobjloader example
            // https://github.com/tinyobjloader/tinyobjloader/blob/release/loader_example.cc
            const std::string basepath = "";
            const std::string& filepath = p["FILEPATH"];

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string warn;
            std::string err;

            // Triangulate obj mesh by default
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(),
                basepath.c_str(), true);

            if (!warn.empty()) {
                std::cout << "WARN: " << warn << std::endl;
            }

            if (!err.empty()) {
                std::cerr << "ERR: " << err << std::endl;
            }

            if (!ret) {
                printf("Failed to load/parse .obj.\n");
                continue;
            }

            // Assume each obj has only one mesh
            newGeom.type = TRIANGLE;
            newGeom.numVertices = 3;
            for (int s = 0; s < shapes.size(); s++) {
                for (int f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                    // Each face is triangulated, loop over each vertex on the face
                    for (int v = 0; v < 3; v++) {
                        // Construct glm::vec3 per vertex
                        tinyobj::index_t meshIndices = shapes[s].mesh.indices[f * 3 + v];
                        int vIndex = static_cast<int>(meshIndices.vertex_index);
                        newGeom.vertices[v] = glm::vec3(
                            attrib.vertices[vIndex * 3 + 0],
                            attrib.vertices[vIndex * 3 + 1],
                            attrib.vertices[vIndex * 3 + 2]
                        );

                        int nIndex = static_cast<int>(meshIndices.normal_index);
                        if (nIndex >= 0) {
                            newGeom.normals[v] = glm::vec3(
                                attrib.normals[nIndex * 3 + 0],
                                attrib.normals[nIndex * 3 + 1],
                                attrib.normals[nIndex * 3 + 2]
                            );
                        }

                        int uvIndex = static_cast<int>(meshIndices.texcoord_index);
                        if (uvIndex >= 0) {
                            newGeom.uv[v] = glm::vec2(
                                attrib.texcoords[uvIndex * 2 + 0],
                                attrib.texcoords[uvIndex * 2 + 1]
                            );
                        }
                    }

                    geoms.push_back(newGeom);
                }
            }
            continue;
        }

        geoms.push_back(newGeom);
    }

    // Assume the scene is static, so we only need to construct BVH once
    int leafSize = 8;
    bvh = BVH(std::move(geoms), leafSize);
    printf("BVH constructed.\n");

    // Copy over reordered geometry data
    geoms = bvh.geoms;
    nodes = bvh.nodes;

    const auto& backgroundData = data["Background"];
    const auto& backGroundType = backgroundData["TYPE"];
    index = textures.size();
    width = 1;
    height = 1;
    if (backGroundType == "CONSTANT") {
        // Constant color environment map
        const auto& backGroundRGB = backgroundData["RGB"];
        textures.push_back(glm::vec4(backGroundRGB[0], backGroundRGB[1], backGroundRGB[2], 1.0f));
    }
    else if (backGroundType == "IMAGE") {
        // Image environment map
        const std::string& filepathImage = backgroundData["FILEPATH"];
        loadImage(filepathImage, index, width, height);
    }
    else {
        // Default black environment map
        textures.push_back(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    bgTextureInfo.index = index;
    bgTextureInfo.width = width;
    bgTextureInfo.height = height;

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.sampleWidth = cameraData.value("SAMPLEWIDTH", 1);
    state.traceDepth = cameraData["DEPTH"];
    size_t fileNameBeignPos = jsonName.find_last_of("/") + 1;
    state.imageName = jsonName.substr(fileNameBeignPos, jsonName.length() - 5 - fileNameBeignPos);
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);
    const auto& focalLength = cameraData.value("FOCALLENGTH", 1.0f);
    const auto& apertureSize = cameraData.value("APERTURESIZE", 0.0f);
    camera.focalLength = focalLength;
    camera.apertureSize = apertureSize;

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
