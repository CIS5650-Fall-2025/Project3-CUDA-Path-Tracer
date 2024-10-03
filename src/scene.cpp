#include <iostream>
#include <cstring>
#include <filesystem>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_obj_loader.h"
#include "stb_image.h"

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

// Use Tiny_OBJ_Loader to read in OBJ mesh file 
// this code is largely from the Example Code (Objected Oriented API) in the Github
// https://github.com/tinyobjloader/tinyobjloader?tab=readme-ov-file#example-code-new-object-oriented-api
void Scene::loadFromObj(std::string path, int idx, Geom& geom)
{
    std::string inputfile = path;
    tinyobj::ObjReaderConfig reader_config;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    geom.meshEnd = 0;
    geom.meshStart = idx;

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            Triangle t;

            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                t.verts[v] = glm::vec3(geom.transform * glm::vec4(vx, vy, vz, 1.0f));

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    t.normals[v] = glm::vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                    t.uvs[v] = glm::vec2(tx, ty);
                }
            }
            index_offset += fv;

            t.geomIdx = geoms.size();
            triangles.push_back(t);

            geom.meshEnd++;
        }
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    int idx = 0;

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

            const float& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - roughness;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            newMaterial.hasRefractive = p["IOR"];
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
        if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else if (type == "cube")
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

        // load OBJ using tiny OBJ loader
        if (newGeom.type == MESH) {

            std::string path{ jsonName };

            // assemble file path
            const std::size_t lastSlashPos{ path.find_last_of('/') };
            path = path.substr(0, lastSlashPos) + std::string("/objs/") + std::string(p["NAME"]) + std::string(".obj");
            std::cout << "OBJ PATH: " << path << std::endl;

            loadFromObj(path, idx, newGeom);
            idx += newGeom.meshEnd;

            // assemble texture path and load
            if (p.contains("TEXTURE")) {
                const std::size_t lastSlashPos{ path.find_last_of('/') };
                path = path.substr(0, lastSlashPos) + std::string("/") + std::string(p["TEXTURE"]);

                newGeom.texIdx = loadTexture(path);
                newGeom.hasTexture = true;

                std::cout << "TEXTURE PATH: " << path << " -- SAVED TO " << newGeom.texIdx << std::endl;
            }
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

    // build bvh
    BuildBVH(5);
    std::cout << "BVH root node: " << rootNodeIdx << std::endl;
    std::cout << "Triangle count: " << triangles.size() << std::endl;
    std::cout << "BVH nodes in tree: " << nodesUsed << std::endl;
    //std::cout << "BVH node size: " << bvhNode.size() << std::endl;
}

// load texture images (using stb_image)
int Scene::loadTexture(std::string path) {

    int width, height, channels;

    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!data) {
        std::cerr << "Failed to load texture: " << path << std::endl;
        return -1;
    }

    // RGBA channels
    channels = 4;

    // texture strut
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.channels = channels;
    texture.data = data;

    int textureId = textures.size();
    textures.push_back(texture);

    return textureId;
}

// reference: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
void Scene::BuildBVH(int maxDepth) {

    std::cout << "Building BVH..." << std::endl;

    // populate triangle index array and compute centroids
    for (int i = 0; i < triangles.size(); i++) {
        triIdx.push_back(i);
        triangles[i].centroid =
            (triangles[i].verts[0] + triangles[i].verts[1] + triangles[i].verts[2]) * 0.3333f;
    }

    // assign all triangles to root node
    BVHNode root;
    root.leftChild = -1;
    root.firstTri = 0;
    root.triCount = triangles.size();
    bvhNode.push_back(root);

    UpdateNodeBounds(rootNodeIdx);
    // subdivide recursively
    Subdivide(rootNodeIdx, 1, maxDepth);

    bvhNode[rootNodeIdx].totalNodes = nodesUsed;
}

void Scene::UpdateNodeBounds(int nodeIdx) {

    BVHNode& node = bvhNode[nodeIdx];

    if (node.triCount > 0) { // for leaf nodes:

        // initialize aabbMin and aabbMax with the first triangle's first vertex
        Triangle& firstTri = triangles[node.firstTri];
        node.aabbMin = firstTri.verts[0];
        node.aabbMax = firstTri.verts[0];

        for (int i = 0; i < node.triCount; ++i) {
            Triangle& tri = triangles[node.firstTri + i];

            // aabb of curr triangle
            glm::vec3 triMin = glm::min(tri.verts[0], glm::min(tri.verts[1], tri.verts[2]));
            glm::vec3 triMax = glm::max(tri.verts[0], glm::max(tri.verts[1], tri.verts[2]));

            // expand node's AABB to include the triangle's AABB
            node.aabbMin = glm::min(node.aabbMin, triMin);
            node.aabbMax = glm::max(node.aabbMax, triMax);
        }
    }
    else { // for internal nodes:

        BVHNode& leftChild = bvhNode[node.leftChild];
        BVHNode& rightChild = bvhNode[node.leftChild + 1];

        // compute AABB by combining children's AABBs
        node.aabbMin = glm::min(leftChild.aabbMin, rightChild.aabbMin);
        node.aabbMax = glm::max(leftChild.aabbMax, rightChild.aabbMax);
    }
}

void Scene::Subdivide(int nodeIdx, int currDepth, int maxDepth) {

    BVHNode& node = bvhNode[nodeIdx];

    // Terminate recursion if the maximum depth is reached or if node has <= 2 triangles
    if (currDepth >= maxDepth || node.triCount <= 2) {
        std::cout << "Depth at End " << currDepth << std::endl;
        return;
    }

    // determine split axis and position
    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

    // in-place partition
    int start = node.firstTri;
    int end = start + node.triCount - 1;

    while (start <= end)
    {
        const glm::vec3& centroid = triangles[triIdx[start]].centroid;
        if (centroid[axis] < splitPos) {
            start++;
        } else {
            std::swap(triIdx[start], triIdx[end]);
            end--;
        }
    }
    // abort split if one of the sides is empty
    int leftCount = start - node.firstTri;
    if (leftCount == 0 || leftCount == node.triCount) return;

    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;

    BVHNode left;
    left.firstTri = node.firstTri;
    left.triCount = leftCount;
    left.leftChild = -1;
    left.aabbMin = glm::vec3(1e30f);
    left.aabbMax = glm::vec3(-1e30f);
    bvhNode.push_back(left);

    BVHNode right;
    right.firstTri = start;
    right.triCount = node.triCount - leftCount;
    right.leftChild = -1;
    right.aabbMin = glm::vec3(1e30f);
    right.aabbMax = glm::vec3(-1e30f);
    bvhNode.push_back(right);
   
    node.leftChild = leftChildIdx;
    node.triCount = 0; // now an internal node, which has no triangles

    UpdateNodeBounds(leftChildIdx);
    UpdateNodeBounds(rightChildIdx);

    // recurse
    Subdivide(leftChildIdx, currDepth + 1, maxDepth);
    Subdivide(rightChildIdx, currDepth + 1, maxDepth);
}
