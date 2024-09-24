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

std::vector<Triangle> Scene::assembleMesh() {
    std::string inputfile = "C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/objs/wolf.obj";
    std::string basestring = "C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/objs/";
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
            newMaterial.specular.isSpecular = false;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.specular.isSpecular = false;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.isSpecular = true;
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
            mesh_triangles = assembleMesh();
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
        bvhNodes.resize(triangle_count);
        triangle_indices.resize(triangle_count);
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
        triangle_indices[i] = i;
        mesh_triangles[i].centroid = (mesh_triangles[i].v0.pos + mesh_triangles[i].v1.pos + mesh_triangles[i].v2.pos) * 0.3333f;
    }
    // assign all triangles to root node
    BVHNode& root = bvhNodes[rootNodeIdx];
    root.leftChild = 0;
    root.firstTriIdx = 0, root.triCount = triangle_count;
    updateNodeBounds(rootNodeIdx);
    // subdivide recursively
    subdivide(rootNodeIdx);
}

glm::vec3 min_vec(glm::vec3 a, glm::vec3 b) {
    return { glm::min(a.x, b.x), glm::min(a.y, b.y) , glm::min(a.z, b.z) };
}

glm::vec3 max_vec(glm::vec3 a, glm::vec3 b) {
    return { glm::max(a.x, b.x), glm::max(a.y, b.y) , glm::max(a.z, b.z) };
}

void Scene::updateNodeBounds(unsigned int nodeIdx)
{
    BVHNode& node = bvhNodes[nodeIdx];
    node.aabbMin = glm::vec3(1e30f);
    node.aabbMax = glm::vec3(-1e30f);
    for (unsigned int first = node.firstTriIdx, i = 0; i < node.triCount; i++)
    {
        unsigned int leafTriIdx = triangle_indices[first + i];
        Triangle& leafTri = mesh_triangles[leafTriIdx];
        node.aabbMin = min_vec(node.aabbMin, leafTri.v0.pos),
        node.aabbMin = min_vec(node.aabbMin, leafTri.v1.pos),
        node.aabbMin = min_vec(node.aabbMin, leafTri.v2.pos),
        node.aabbMax = max_vec(node.aabbMax, leafTri.v0.pos),
        node.aabbMax = max_vec(node.aabbMax, leafTri.v1.pos),
        node.aabbMax = max_vec(node.aabbMax, leafTri.v2.pos);
    }
}

void Scene::subdivide(unsigned int nodeIdx) {
    // terminate recursion
    BVHNode& node = bvhNodes[nodeIdx];
    if (node.triCount <= 2) return;
    // determine split axis and position
    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
    // in-place partition
    int i = node.firstTriIdx;
    int j = i + node.triCount - 1;
    while (i <= j)
    {
        if (mesh_triangles[triangle_indices[i]].centroid[axis] < splitPos)
            i++;
        else
            swap(triangle_indices[i], triangle_indices[j--]);
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.firstTriIdx;
    if (leftCount == 0 || leftCount == node.triCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    bvhNodes[leftChildIdx].firstTriIdx = node.firstTriIdx;
    bvhNodes[leftChildIdx].triCount = leftCount;
    bvhNodes[rightChildIdx].firstTriIdx = i;
    bvhNodes[rightChildIdx].triCount = node.triCount - leftCount;
    node.leftChild = leftChildIdx;
    node.triCount = 0;
    updateNodeBounds(leftChildIdx);
    updateNodeBounds(rightChildIdx);
    // recurse
    subdivide(leftChildIdx);
    subdivide(rightChildIdx);
}

void IntersectTri(Ray & ray, const Triangle & tri)
{
    const glm::vec3 edge1 = tri.v1.pos - tri.v0.pos;
    const glm::vec3 edge2 = tri.v2.pos - tri.v0.pos;
    const glm::vec3 h = glm::cross(ray.direction, edge2);
    const float a = dot(edge1, h);
    if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
    const float f = 1 / a;
    const glm::vec3 s = ray.origin - tri.v0.pos;
    const float u = f * glm::dot(s, h);
    if (u < 0 || u > 1) return;
    const glm::vec3 q = glm::cross(s, edge1);
    const float v = f * glm::dot(ray.direction, q);
    if (v < 0 || u + v > 1) return;
    const float t = f * glm::dot(edge2, q);
    if (t > 0.0001f) ray.t = min(ray.t, t);
}

void Scene::intersectBVH(Ray& ray, const unsigned int nodeIdx) {
    BVHNode& node = bvhNodes[nodeIdx];
    if (!intersectAABB(ray, node.aabbMin, node.aabbMax)) {
        return;
    }
    if (node.isLeaf())
    {
        for (unsigned int i = 0; i < node.triCount; i++) {
            IntersectTri(ray, mesh_triangles[triangle_indices[node.firstTriIdx + i]]);
        }
    }
    else
    {
        intersectBVH(ray, node.leftChild);
        intersectBVH(ray, node.leftChild + 1);
    }
}

bool Scene::intersectAABB(const Ray & ray, const glm::vec3 bmin, const glm::vec3 bmax) {
    glm::vec3 ro = ray.origin, rd = ray.direction;
    float tx1 = (bmin.x - ro.x) / rd.x, tx2 = (bmax.x - ro.x) / rd.x;
    float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
    float ty1 = (bmin.y - ro.y) / rd.y, ty2 = (bmax.y - ro.y) / rd.y;
    tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
    float tz1 = (bmin.z - ro.z) / rd.z, tz2 = (bmax.z - ro.z) / rd.z;
    tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
    return tmax >= tmin && tmin < ray.t && tmax > 0;
}
