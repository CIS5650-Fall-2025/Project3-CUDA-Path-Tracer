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

#if BVH == 1
                t.verts[v] = glm::vec3(geom.transform * glm::vec4(vx, vy, vz, 1.0f));
#else
                t.verts[v] = glm::vec3(vx, vy, vz);
#endif

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

            float R0 = (1.0f - p["IOR"]) / (1.0f + p["IOR"]);
            newMaterial.R0sq = R0 * R0;
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

                newGeom.texIdx = loadTexture(path, std::string(p["TEXTURE"]));
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

#if BVH == 1
    // build bvh
    buildBVH(BVHDEPTH);
    std::cout << "BVH root node: " << rootNodeIdx << std::endl;
    std::cout << "Triangle count: " << triangles.size() << std::endl;
    std::cout << "BVH nodes in tree: " << nodesUsed << std::endl;
    //std::cout << "BVH node size: " << bvhNode.size() << std::endl;
#endif
}

// load texture images (using stb_image)
int Scene::loadTexture(std::string path, std::string name) {

    if (name == "PROCEDURAL1") {
        return -2;
    } 
    else if (name == "PROCEDURAL2") {
        return -3;
    }
    else {

        int width, height, channels;
        const std::size_t lastSlashPos{ path.find_last_of('/') };
        path = path.substr(0, lastSlashPos) + std::string("/") + name;

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
}

// reference for all BVH functions: 
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

void Scene::buildBVH(int maxDepth) {

    std::cout << "Building BVH..." << std::endl;

    // populate triangle index array and compute centroids
    for (int i = 0; i < triangles.size(); i++) {
        triIdx.push_back(i);
        triangles[i].centroid =
            (triangles[i].verts[0] + triangles[i].verts[1] + triangles[i].verts[2]) * 0.3333f;
    }

    // assign all triangles to root node
    BVHNode root;
    root.leftFirst = 0;
    root.triCount = triangles.size();
    bvhNode.push_back(root);

    updateNodeBounds(rootNodeIdx);
    // subdivide recursively
    subdivide(rootNodeIdx, 1, maxDepth);

    bvhNode[rootNodeIdx].totalNodes = nodesUsed;
    std::cout << "Finish BVH" << std::endl;
}

void Scene::updateNodeBounds(int nodeIdx) {

    BVHNode& node = bvhNode[nodeIdx];
    
    if (node.triCount > 0) { // for leaf nodes:

        // initialize aabbMin and aabbMax with the first triangle's first vertex
        Triangle& leftFirst = triangles[node.leftFirst];
        node.aabbMin = leftFirst.verts[0];
        node.aabbMax = leftFirst.verts[0];

        for (int i = 0; i < node.triCount; ++i) {
            Triangle& tri = triangles[node.leftFirst + i];

            // aabb of curr triangle
            glm::vec3 triMin = glm::min(tri.verts[0], glm::min(tri.verts[1], tri.verts[2]));
            glm::vec3 triMax = glm::max(tri.verts[0], glm::max(tri.verts[1], tri.verts[2]));

            // expand node's AABB to include the triangle's AABB
            node.aabbMin = glm::min(node.aabbMin, triMin);
            node.aabbMax = glm::max(node.aabbMax, triMax);
        }
    }
    else { // for internal nodes:

        BVHNode& leftFirst = bvhNode[node.leftFirst];
        BVHNode& rightChild = bvhNode[node.leftFirst + 1];

        // compute AABB by combining children's AABBs
        node.aabbMin = glm::min(leftFirst.aabbMin, rightChild.aabbMin);
        node.aabbMax = glm::max(leftFirst.aabbMax, rightChild.aabbMax);
    }

}

void Scene::subdivide(int nodeIdx, int currDepth, int maxDepth) {

    BVHNode& node = bvhNode[nodeIdx];

    // terminate recursion if the maximum depth is reached or if node has <= 2 triangles
    if (currDepth >= maxDepth || node.triCount <= 2) {
        //std::cout << "Depth at End " << currDepth << std::endl;
        return;
    }

    // determine split axis using SAH
    int axis;
    float splitPos;

    float nosplitCost = node.cost();

    // splitting doesn't reduce costs, leave as is
    if (splitPlane(node, axis, splitPos) >= nosplitCost) return;

    // partition triangles in-place based on best split
    int start = node.leftFirst;
    int end = start + node.triCount - 1;

    while (start <= end)
    {
        const glm::vec3& centroid = triangles[triIdx[start]].centroid;
        if (centroid[axis] < splitPos) {
            start++;
        }
        else {
            std::swap(triIdx[start], triIdx[end]);
            end--;
        }
    }

    int leftCount = start - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount) return;

    // create child nodes
    int leftFirstIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;

    BVHNode left;
    left.leftFirst = node.leftFirst;
    left.triCount = leftCount;
    left.aabbMin = glm::vec3(1e30f);
    left.aabbMax = glm::vec3(-1e30f);
    bvhNode.push_back(left);

    BVHNode right;
    right.leftFirst = start;
    right.triCount = node.triCount - leftCount;
    right.aabbMin = glm::vec3(1e30f);
    right.aabbMax = glm::vec3(-1e30f);
    bvhNode.push_back(right);

    node.leftFirst = leftFirstIdx;
    node.triCount = 0; // now an internal node, which has no triangles

    updateNodeBounds(leftFirstIdx);
    updateNodeBounds(rightChildIdx);
    std::cout << "subbing" << std::endl;
    // recurse
    subdivide(leftFirstIdx, currDepth + 1, maxDepth);
    subdivide(rightChildIdx, currDepth + 1, maxDepth);
}

float Scene::splitPlane(BVHNode& node, int& bestAxis, float& splitPos)
{
    float bestCost = 1e30f;

    for (int axis = 0; axis < 3; axis++)
    {
        float boundsMin = 1e30f;
        float boundsMax = -1e30f;

        for (int i = 0; i < node.triCount; i++)
        {
            const Triangle& triangle = triangles[triIdx[node.leftFirst + i]];
            boundsMin = min(boundsMin, triangle.centroid[axis]);
            boundsMax = max(boundsMax, triangle.centroid[axis]);
        }

        if (boundsMin == boundsMax) continue;

        std::array<Bin, BINS> bin;
        float scale = BINS / (boundsMax - boundsMin);

        // assign each triangle to bin based on centroid location
        for (int i = 0; i < node.triCount; i++)
        {
            Triangle& triangle = triangles[triIdx[node.leftFirst + i]];
            int binIdx = static_cast<int>((triangle.centroid[axis] - boundsMin) * scale);
            binIdx = std::min(binIdx, BINS - 1); // clamp to last bin

            bin[binIdx].bounds.grow(triangle.verts[0]);
            bin[binIdx].bounds.grow(triangle.verts[1]);
            bin[binIdx].bounds.grow(triangle.verts[2]);

            bin[binIdx].triCount++;
        }

        // prefix sums for SAH
        std::array<float, BINS - 1> leftArea;
        std::array<int, BINS - 1> leftCount;
        BVHNode left;

        for (int i = 0; i < BINS - 1; ++i) {
            left.grow(bin[i].bounds.aabbMin);
            left.grow(bin[i].bounds.aabbMax);
            leftCount[i] = (i == 0) ? bin[i].triCount : leftCount[i - 1] + bin[i].triCount;
            leftArea[i] = left.area();
        }

        // suffix sums for SAH
        std::array<float, BINS - 1> rightArea;
        std::array<int, BINS - 1> rightCount;
        BVHNode right;

        for (int i = BINS - 1; i >= 1; --i) {
            right.grow(bin[BINS - 1 - i].bounds.aabbMin);
            right.grow(bin[BINS - 1 - i].bounds.aabbMax);
            int idx = i - 1;
            rightCount[idx] = (i == BINS - 1) ? bin[i].triCount : rightCount[idx + 1] + bin[i].triCount;
            rightArea[idx] = right.area();
        }

        // evaluate SAH cost for each potential split
        float binWidth = (boundsMax - boundsMin) / BINS;
        for (int i = 0; i < BINS - 1; ++i) {

            if (leftCount[i] == 0 || rightCount[i] == 0)
                continue;

            float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
            if (planeCost < bestCost) {
                bestCost = planeCost;
                bestAxis = axis;
                splitPos = boundsMin + binWidth * (i + 1);
            }
        }
    }
    return bestCost;
}