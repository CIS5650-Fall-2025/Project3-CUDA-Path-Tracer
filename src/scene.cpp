#include "scene.h"

#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define TINYGLTF_IMPLEMENTATION
#include "json.hpp"
#include "tiny_gltf.h"
using json = nlohmann::json;

Scene::Scene(string filename) {
  cout << "Reading scene from " << filename << " ..." << endl;
  cout << " " << endl;
  auto ext = filename.substr(filename.find_last_of('.'));
  if (ext == ".json") {
    loadFromJSON(filename);
    return;
  } else {
    cout << "Couldn't read from " << filename << endl;
    exit(-1);
  }
}

void Scene::loadFromJSON(const std::string& jsonName) {
  std::ifstream f(jsonName);
  json data = json::parse(f);
  const auto& materialsData = data["Materials"];
  std::unordered_map<std::string, uint32_t> MatNameToID;
  for (const auto& item : materialsData.items()) {
    const auto& name = item.key();
    const auto& p = item.value();
    Material newMaterial{};
    const auto& col = p["RGB"];
    newMaterial.color = glm::vec3(col[0], col[1], col[2]);
    newMaterial.hasReflective = false;
    newMaterial.hasRefractive = false;
    newMaterial.indexOfRefraction = 0.0f;
    if (p["TYPE"] == "Diffuse") {
      // Diffuse material
      newMaterial.emittance = 0.0f;
    } else if (p["TYPE"] == "Emitting") {
      // Emissive material
      newMaterial.emittance = p["EMITTANCE"];
    } else if (p["TYPE"] == "Specular") {
      // Specular (mirror-like) material
      newMaterial.hasReflective = true;
      newMaterial.emittance = 0.0f;
      newMaterial.specular.exponent = p["ROUGHNESS"];
      newMaterial.specular.color = newMaterial.color;
    } else if (p["TYPE"] == "Refractive") {
      // Refractive (glass) material
      newMaterial.hasRefractive = true;
      newMaterial.emittance = 0.0f;
      newMaterial.indexOfRefraction = p["IOR"];
      newMaterial.specular.exponent = p["ROUGHNESS"];
      newMaterial.specular.color = newMaterial.color;
    }
    MatNameToID[name] = materials.size();
    materials.emplace_back(newMaterial);
  }
  const auto& objectsData = data["Objects"];
  std::string path = jsonName.substr(0, min(jsonName.find_last_of('\\'), jsonName.find_last_of('/')) + 1);
  for (const auto& p : objectsData) {
    const auto& type = p["TYPE"];
    const auto& trans = p["TRANS"];
    const auto& rotat = p["ROTAT"];
    const auto& scal = p["SCALE"];
    glm::vec3 translate(trans[0], trans[1], trans[2]);
    glm::vec3 rotate(rotat[0], rotat[1], rotat[2]);
    glm::vec3 scale(scal[0], scal[1], scal[2]);
    if (type == "mesh") {
      std::string filename = p["FILE"].get<std::string>();
      loadMeshesFromGLTF(path + filename, geoms, triangles, materials, translate, rotate, scale);
    } else {
      Geom newGeom;
      if (type == "cube") {
        newGeom.type = CUBE;
      } else {
        newGeom.type = SPHERE;
      }
      newGeom.materialid = MatNameToID[p["MATERIAL"]];
      newGeom.translation = translate;
      newGeom.rotation = rotate;
      newGeom.scale = scale;
      newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
      newGeom.inverseTransform = glm::inverse(newGeom.transform);
      newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

      geoms.push_back(newGeom);
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

  camera.lensRadius = cameraData.contains("LENSRADIUS") ? cameraData["LENSRADIUS"] : 0.0f;
  camera.focalDistance = cameraData.contains("FOCALDISTANCE") ? cameraData["FOCALDISTANCE"] : 1.0f;

  // calculate fov based on resolution
  float yscaled = tan(fovy * (PI / 180));
  float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(xscaled) * 180) / PI;
  camera.fov = glm::vec2(fovx, fovy);

  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);

  camera.view = glm::normalize(camera.lookAt - camera.position);

  // set up render camera stuff
  int arraylen = camera.resolution.x * camera.resolution.y;
  state.image.resize(arraylen);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

glm::mat4 getLocalTransform(const tinygltf::Node& node) {
  glm::mat4 matrix(1.0f);
  if (!node.matrix.empty()) {
    for (int i = 0; i < 16; ++i) {
      matrix[i / 4][i % 4] = static_cast<float>(node.matrix[i]);
    }
  } else {
    glm::vec3 translation(0.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale(1.0f);

    if (!node.translation.empty()) {
      translation = glm::vec3(static_cast<float>(node.translation[0]), static_cast<float>(node.translation[1]),
                              static_cast<float>(node.translation[2]));
    }
    if (!node.rotation.empty()) {
      rotation = glm::quat(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                           static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]));
    }
    if (!node.scale.empty()) {
      scale = glm::vec3(static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]),
                        static_cast<float>(node.scale[2]));
    }

    matrix =
        glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
  }
  return matrix;
}

void loadIndices(const tinygltf::Model& model, int accessorIndex, std::vector<unsigned int>& outIndices) {
  if (accessorIndex < 0) return;
  const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
  const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

  const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  size_t indexCount = accessor.count;

  outIndices.resize(indexCount);

  for (size_t i = 0; i < indexCount; ++i) {
    unsigned int index = 0;
    switch (accessor.componentType) {
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        index = static_cast<unsigned int>(*(reinterpret_cast<const uint8_t*>(dataPtr + i * sizeof(uint8_t))));
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        index = static_cast<unsigned int>(*(reinterpret_cast<const uint16_t*>(dataPtr + i * sizeof(uint16_t))));
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        index = *(reinterpret_cast<const uint32_t*>(dataPtr + i * sizeof(uint32_t)));
        break;
      default:
        std::cerr << "Unsupported index component type." << std::endl;
        break;
    }
    outIndices[i] = index;
  }
}

void loadPositions(const tinygltf::Model& model, const tinygltf::Primitive& primitive,
                   std::vector<glm::vec3>& outPositions) {
  auto it = primitive.attributes.find("POSITION");
  if (it == primitive.attributes.end()) {
    std::cerr << "Mesh primitive has no POSITION attribute." << std::endl;
    return;
  }
  const tinygltf::Accessor& accessor = model.accessors[it->second];
  const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

  const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  size_t vertexCount = accessor.count;

  size_t byteStride = accessor.ByteStride(bufferView);
  if (byteStride == 0) {
    byteStride = 3 * sizeof(float);
  }

  outPositions.resize(vertexCount);

  for (size_t i = 0; i < vertexCount; ++i) {
    const float* elem = reinterpret_cast<const float*>(dataPtr + i * byteStride);
    outPositions[i] = glm::vec3(elem[0], elem[1], elem[2]);
  }
}

void loadNormals(const tinygltf::Model& model, const tinygltf::Primitive& primitive,
                 std::vector<glm::vec3>& outNormals) {
  auto it = primitive.attributes.find("NORMAL");
  if (it == primitive.attributes.end()) {
    return;
  }
  const tinygltf::Accessor& accessor = model.accessors[it->second];
  const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

  const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  size_t vertexCount = accessor.count;

  size_t byteStride = accessor.ByteStride(bufferView);
  if (byteStride == 0) {
    byteStride = 3 * sizeof(float);
  }

  outNormals.resize(vertexCount);

  for (size_t i = 0; i < vertexCount; ++i) {
    const float* elem = reinterpret_cast<const float*>(dataPtr + i * byteStride);
    outNormals[i] = glm::vec3(elem[0], elem[1], elem[2]);
  }
}

void loadTexcoords(const tinygltf::Model& model, const tinygltf::Primitive& primitive,
                   std::vector<glm::vec2>& outTexcoords) {
  auto it = primitive.attributes.find("TEXCOORD_0");
  if (it == primitive.attributes.end()) {
    return;
  }
  const tinygltf::Accessor& accessor = model.accessors[it->second];
  const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

  const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  size_t vertexCount = accessor.count;

  size_t byteStride = accessor.ByteStride(bufferView);
  if (byteStride == 0) {
    byteStride = 2 * sizeof(float);
  }

  outTexcoords.resize(vertexCount);

  for (size_t i = 0; i < vertexCount; ++i) {
    const float* elem = reinterpret_cast<const float*>(dataPtr + i * byteStride);
    outTexcoords[i] = glm::vec2(elem[0], elem[1]);
  }
}

int mapMaterial(int gltfMaterialIndex, std::vector<Material>& materials, std::unordered_map<int, int>& materialIdMap,
                const tinygltf::Model& model) {
  if (gltfMaterialIndex < 0) {
    return 0;
  }

  auto it = materialIdMap.find(gltfMaterialIndex);
  if (it != materialIdMap.end()) {
    return it->second;
  }

  const tinygltf::Material& gltfMaterial = model.materials[gltfMaterialIndex];
  Material mat;
  mat.color = glm::vec3(1.0f);
  mat.specular.color = glm::vec3(0.0f);
  mat.hasReflective = false;
  mat.hasRefractive = false;
  mat.indexOfRefraction = 1.0f;
  mat.emittance = 0.0f;

  if (!gltfMaterial.pbrMetallicRoughness.baseColorFactor.empty()) {
    mat.color = glm::vec3(static_cast<float>(gltfMaterial.pbrMetallicRoughness.baseColorFactor[0]),
                          static_cast<float>(gltfMaterial.pbrMetallicRoughness.baseColorFactor[1]),
                          static_cast<float>(gltfMaterial.pbrMetallicRoughness.baseColorFactor[2]));
  }

  double metallicFactor = gltfMaterial.pbrMetallicRoughness.metallicFactor;
  double roughnessFactor = gltfMaterial.pbrMetallicRoughness.roughnessFactor;

  if (metallicFactor > 0.5) {
    mat.hasReflective = true;
    mat.specular.color = mat.color;
  }

  if (!gltfMaterial.emissiveFactor.empty()) {
    mat.emittance = static_cast<float>(gltfMaterial.emissiveFactor[0]) +
                    static_cast<float>(gltfMaterial.emissiveFactor[1]) +
                    static_cast<float>(gltfMaterial.emissiveFactor[2]);
    if (mat.emittance > 0.0f) {
      mat.color = glm::vec3(static_cast<float>(gltfMaterial.emissiveFactor[0]),
                            static_cast<float>(gltfMaterial.emissiveFactor[1]),
                            static_cast<float>(gltfMaterial.emissiveFactor[2]));
    }
  }

  int materialId = static_cast<int>(materials.size());
  materials.push_back(mat);
  materialIdMap[gltfMaterialIndex] = materialId;
  return materialId;
}

void loadMaterialsFromGLTF(const tinygltf::Model& model, std::vector<Material>& materials) {
  for (const tinygltf::Material& gltfMaterial : model.materials) {
    Material mat;
    mat.color = glm::vec3(1.0f);
    mat.specular.color = glm::vec3(0.0f);
    mat.hasReflective = false;
    mat.hasRefractive = false;
    mat.indexOfRefraction = 1.0f;
    mat.emittance = 0.0f;

    if (gltfMaterial.values.find("baseColorFactor") != gltfMaterial.values.end()) {
      const tinygltf::ColorValue& color = gltfMaterial.values.at("baseColorFactor").ColorFactor();
      mat.color = glm::vec3(color[0], color[1], color[2]);
    }

    if (gltfMaterial.values.find("metallicFactor") != gltfMaterial.values.end()) {
      double metallic = gltfMaterial.values.at("metallicFactor").Factor();
      if (metallic > 0.5) {
        mat.hasReflective = true;
        mat.specular.color = mat.color;
      }
    }

    if (gltfMaterial.additionalValues.find("emissiveFactor") != gltfMaterial.additionalValues.end()) {
      const tinygltf::ColorValue& emissive = gltfMaterial.additionalValues.at("emissiveFactor").ColorFactor();
      mat.emittance = glm::length(glm::vec3(emissive[0], emissive[1], emissive[2]));
    }

    materials.push_back(mat);
  }
}

void loadNode(const tinygltf::Model& model, const tinygltf::Node& node, const glm::mat4& parentTransform,
              std::vector<Geom>& geoms, std::vector<Triangle>& triangles, std::vector<Material>& materials,
              std::unordered_map<int, int>& materialIdMap) {
  glm::mat4 localTransform = parentTransform * getLocalTransform(node);

  if (node.mesh >= 0) {
    const tinygltf::Mesh& mesh = model.meshes[node.mesh];
    Geom geom;
    geom.type = MESH;
    geom.transform = localTransform;
    geom.inverseTransform = glm::inverse(localTransform);
    geom.invTranspose = glm::inverseTranspose(localTransform);

    geom.meshTriStartIdx = triangles.size();
    int totalTriangles = 0;

    for (const tinygltf::Primitive& primitive : mesh.primitives) {
      std::vector<unsigned int> indices;
      loadIndices(model, primitive.indices, indices);

      std::vector<glm::vec3> positions;
      std::vector<glm::vec3> normals;
      std::vector<glm::vec2> texcoords;

      loadPositions(model, primitive, positions);
      loadNormals(model, primitive, normals);
      loadTexcoords(model, primitive, texcoords);

      int materialId = mapMaterial(primitive.material, materials, materialIdMap, model);

      size_t triangleCount = indices.size() / 3;
      for (size_t t = 0; t < triangleCount; ++t) {
        Triangle tri;
        unsigned int idx0 = indices[t * 3 + 0];
        unsigned int idx1 = indices[t * 3 + 1];
        unsigned int idx2 = indices[t * 3 + 2];

        tri.vertices[0] = positions[idx0];
        tri.vertices[1] = positions[idx1];
        tri.vertices[2] = positions[idx2];

        tri.normals[0] = normals.empty() ? glm::vec3(0.0f, 1.0f, 0.0f) : normals[idx0];
        tri.normals[1] = normals.empty() ? glm::vec3(0.0f, 1.0f, 0.0f) : normals[idx1];
        tri.normals[2] = normals.empty() ? glm::vec3(0.0f, 1.0f, 0.0f) : normals[idx2];

        tri.uvs[0] = texcoords.empty() ? glm::vec2(0.0f) : texcoords[idx0];
        tri.uvs[1] = texcoords.empty() ? glm::vec2(0.0f) : texcoords[idx1];
        tri.uvs[2] = texcoords.empty() ? glm::vec2(0.0f) : texcoords[idx2];

        tri.materialId = materialId;

        triangles.push_back(tri);
      }
      totalTriangles += static_cast<int>(triangleCount);
    }

    geom.meshTriCount = totalTriangles;
    geoms.push_back(geom);
  }

  for (int childIdx : node.children) {
    const tinygltf::Node& childNode = model.nodes[childIdx];
    loadNode(model, childNode, localTransform, geoms, triangles, materials, materialIdMap);
  }
}

void Scene::loadMeshesFromGLTF(const std::string& filename, std::vector<Geom>& geoms, std::vector<Triangle>& triangles,
                               std::vector<Material>& materials) {
  glm::vec3 translate(0.0f);
  glm::vec3 rotate(0.0f);
  glm::vec3 scale(1.0f);
  loadMeshesFromGLTF(filename, geoms, triangles, materials, translate, rotate, scale);
}

void Scene::loadMeshesFromGLTF(const std::string& filename, std::vector<Geom>& geoms, std::vector<Triangle>& triangles,
                               std::vector<Material>& materials, const glm::vec3& translate, const glm::vec3& rotate,
                               const glm::vec3& scale) {
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  auto ext = filename.substr(filename.find_last_of('.'));
  bool ret;
  if (ext == ".gltf") {
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
  } else if (ext == ".glb") {
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
  }

  if (!warn.empty()) {
    std::cout << "glTF Warning: " << warn << std::endl;
  }
  if (!err.empty()) {
    std::cerr << "glTF Error: " << err << std::endl;
  }
  if (!ret) {
    std::cerr << "Failed to load glTF file: " << filename << std::endl;
    return;
  }

  loadMaterialsFromGLTF(model, materials);
  std::unordered_map<int, int> materialIdMap;  // glTF material index to custom material index

  glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), translate);
  glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), rotate.x, glm::vec3(1.0f, 0.0f, 0.0f));
  rotationMatrix = glm::rotate(rotationMatrix, rotate.y, glm::vec3(0.0f, 1.0f, 0.0f));
  rotationMatrix = glm::rotate(rotationMatrix, rotate.z, glm::vec3(0.0f, 0.0f, 1.0f));
  glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
  glm::mat4 parentTransform = translationMatrix * rotationMatrix * scaleMatrix;

  for (const tinygltf::Scene& scene : model.scenes) {
    for (int nodeIdx : scene.nodes) {
      const tinygltf::Node& node = model.nodes[nodeIdx];
      loadNode(model, node, parentTransform, geoms, triangles, materials, materialIdMap);
    }
  }
}
