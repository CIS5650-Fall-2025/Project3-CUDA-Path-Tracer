#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation. Requires C++11
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

#include "json.hpp"
#include "scene.h"
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

std::vector<Triangle> Scene::assembleMesh(std::string& inputfile, std::string& basestring) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str(), basestring.c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    std::vector<Triangle> mesh_tris;

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            Triangle new_tri;

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                Vertex& current_vert = (v == 0 ? new_tri.v0 : (v == 1) ? new_tri.v1 : new_tri.v2);

                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                current_vert.pos = glm::vec3(vx, vy, vz);

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    current_vert.nor = glm::vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    current_vert.uv = glm::vec2(tx, ty);
                }
            }

            mesh_tris.push_back(new_tri);

            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

    return mesh_tris;
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
            newMaterial.specular_transmissive.isSpecular = false;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.specular_transmissive.isSpecular = false;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular_transmissive.isSpecular = true;
        }
        else if (p["TYPE"] == "SpecularTransmissive") 
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular_transmissive.isSpecular = true;
            newMaterial.specular_transmissive.isTransmissive = true;

            const auto& eta = p["ETA"];
            newMaterial.specular_transmissive.eta = glm::vec2(eta[0], eta[1]);
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
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "triangle") {
            newGeom.type = TRIANGLE;

            const auto& p0 = p["p0"];
            const auto& p1 = p["p1"];
            const auto& p2 = p["p2"];

            Triangle new_tri;

            new_tri.v0.pos = glm::vec3(p0[0], p0[1], p0[2]);
            new_tri.v1.pos = glm::vec3(p1[0], p1[1], p1[2]);
            new_tri.v2.pos = glm::vec3(p2[0], p2[1], p2[2]);

            newGeom.tris[0] = new_tri;
        }
        else if (type == "mesh") {
            const auto& file = p["FILE"];
            const auto& file_folder = p["FILE_FOLDER"];
            std::string file_str = file;
            std::string file_folder_str = file_folder;
            mesh_triangles = assembleMesh(file_str, file_folder_str);
            triangle_count = mesh_triangles.size();
            newGeom.tris = mesh_triangles.data();
            newGeom.type = MESH;
            meshes.push_back(newGeom);
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

    //build BVH
    if (meshes.size() > 0) {
        bvhNodes.resize(triangle_count * 2 - 1);
        constructBVH();
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

//BVH CONSTRUCTION

void Scene::constructBVH() {
    for (int i = 0; i < triangle_count; i++) {
        mesh_triangles[i].centroid = (mesh_triangles[i].v0.pos + mesh_triangles[i].v1.pos + mesh_triangles[i].v2.pos) * 0.3333f;
    }
    // assign all triangles to root node
    BVHNode& root = bvhNodes[rootNodeIdx];
    root.leftFirst = 0, root.triCount = triangle_count;
    updateNodeBounds(root);
    // subdivide recursively
    subdivide(root);
}

glm::vec3 min_vec(glm::vec3 a, glm::vec3 b) {
    return { glm::min(a.x, b.x), glm::min(a.y, b.y) , glm::min(a.z, b.z) };
}

glm::vec3 max_vec(glm::vec3 a, glm::vec3 b) {
    return { glm::max(a.x, b.x), glm::max(a.y, b.y) , glm::max(a.z, b.z) };
}

void Scene::updateNodeBounds(BVHNode& node)
{
    for (int i = 0; i < node.triCount; ++i)
    {
        const Triangle& leafTri = mesh_triangles[node.leftFirst + i];
        node.aabb.grow(leafTri.v0.pos);
        node.aabb.grow(leafTri.v1.pos);
        node.aabb.grow(leafTri.v2.pos);
    }
}

float Scene::evaluateSAH(BVHNode& node, int axis, float pos)
{
    // determine triangle counts and bounds for this split candidate
    bbox leftBox, rightBox;
    int leftCount = 0, rightCount = 0;
    for (unsigned int i = 0; i < node.triCount; i++)
    {
        Triangle& triangle = mesh_triangles[node.leftFirst + i];
        if (triangle.centroid[axis] < pos)
        {
            leftCount++;
            leftBox.grow(triangle.v0.pos);
            leftBox.grow(triangle.v1.pos);
            leftBox.grow(triangle.v2.pos);
        }
        else
        {
            rightCount++;
            rightBox.grow(triangle.v0.pos);
            rightBox.grow(triangle.v1.pos);
            rightBox.grow(triangle.v2.pos);
        }
    }
    float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return cost > 0 ? cost : 1e30f;
}

void Scene::subdivide(BVHNode& node) {
    // terminate recursion
    if (node.triCount <= 2) return;

    // determine split axis using SAH
    int bestAxis = -1;
    float bestPos = 0, bestCost = 1e30f;
    for (int axis = 0; axis < 3; axis++) {
        for (unsigned int i = 0; i < node.triCount; i++) {
            Triangle& triangle = mesh_triangles[node.leftFirst + i];
            float candidatePos = triangle.centroid[axis];
            float cost = evaluateSAH(node, axis, candidatePos);
            if (cost < bestCost)
                bestPos = candidatePos, bestAxis = axis, bestCost = cost;
        }
    }
    int axis = bestAxis;
    float splitPos = bestPos;

    //check if it is worth it to split
    float parentArea = node.aabb.area();
    float parentCost = node.triCount * parentArea;
    if (bestCost >= parentCost) return;

    // in-place partition
    int i = node.leftFirst;
    int j = i + node.triCount - 1;
    while (i <= j)
    {
        if (mesh_triangles[i].centroid[axis] < splitPos)
            i++;
        else
            swap(mesh_triangles[i], mesh_triangles[j--]);
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;

    BVHNode& leftChild = bvhNodes[leftChildIdx];
    leftChild.leftFirst = node.leftFirst;
    leftChild.triCount = leftCount;
    leftChild.aabb = bbox();

    BVHNode& rightChild = bvhNodes[rightChildIdx];
    rightChild.leftFirst = i;
    rightChild.triCount = node.triCount - leftCount;
    rightChild.aabb = bbox();

    node.leftFirst = leftChildIdx;
    node.triCount = 0;
    updateNodeBounds(leftChild);
    updateNodeBounds(rightChild);

    // recurse
    subdivide(leftChild);
    subdivide(rightChild);
}
