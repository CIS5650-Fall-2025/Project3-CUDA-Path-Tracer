#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"

using namespace std;

using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        constructBVHTree();
        //drawBoundingBox(nodes[0].mins, nodes[1].maxs, glm::vec3(1.f));
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
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 0.f;
            newMaterial.indexOfRefraction = 1.55f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
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

float getMin(float x, float y, float z)
{
    return fminf(fminf(x, y), fminf(y, z));
}

float getMax(float x, float y, float z)
{
    return fmaxf(fmaxf(x, y), fmaxf(y, z));
}

void getTriangleAABB(Geom& tri, glm::vec3& mins, glm::vec3& maxs)
{
    glm::vec4 vert0 = tri.transform * glm::vec4(tri.verts[0], 1.f);
    glm::vec4 vert1 = tri.transform * glm::vec4(tri.verts[1], 1.f);
    glm::vec4 vert2 = tri.transform * glm::vec4(tri.verts[2], 1.f);
    mins.x = getMin(vert0.x, vert1.x, vert2.x);
    mins.y = getMin(vert0.y, vert1.y, vert2.y);
    mins.z = getMin(vert0.z, vert1.z, vert2.z);
    maxs.x = getMax(vert0.x, vert1.x, vert2.x);
    maxs.y = getMax(vert0.y, vert1.y, vert2.y);
    maxs.z = getMax(vert0.z, vert1.z, vert2.z);
}

void getOtherPrimitiveAABB(Geom& geom, glm::vec3& mins, glm::vec3& maxs)
{
    glm::vec4 corner0 = geom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f);
    glm::vec4 corner1 = geom.transform * glm::vec4(-0.5f, -0.5f, 0.5f, 1.f);
    glm::vec4 corner2 = geom.transform * glm::vec4(-0.5f, 0.5f, -0.5f, 1.f);
    glm::vec4 corner3 = geom.transform * glm::vec4(0.5f, -0.5f, -0.5f, 1.f);
    glm::vec4 corner4 = geom.transform * glm::vec4(-0.5f, 0.5f, 0.5f, 1.f);
    glm::vec4 corner5 = geom.transform * glm::vec4(0.5f, -0.5f, 0.5f, 1.f);
    glm::vec4 corner6 = geom.transform * glm::vec4(0.5f, 0.5f, -0.5f, 1.f);
    glm::vec4 corner7 = geom.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.f);
    mins.x = getMin(getMin(corner0.x, corner1.x, corner2.x), getMin(corner3.x, corner4.x, corner5.x), fmin(corner6.x, corner7.x));
    mins.y = getMin(getMin(corner0.y, corner1.y, corner2.y), getMin(corner3.y, corner4.y, corner5.y), fmin(corner6.y, corner7.y));
    mins.z = getMin(getMin(corner0.z, corner1.z, corner2.z), getMin(corner3.z, corner4.z, corner5.z), fmin(corner6.z, corner7.z));
    maxs.x = getMax(getMax(corner0.x, corner1.x, corner2.x), getMax(corner3.x, corner4.x, corner5.x), fmax(corner6.x, corner7.x));
    maxs.y = getMax(getMax(corner0.y, corner1.y, corner2.y), getMax(corner3.y, corner4.y, corner5.y), fmax(corner6.y, corner7.y));
    maxs.z = getMax(getMax(corner0.z, corner1.z, corner2.z), getMax(corner3.z, corner4.z, corner5.z), fmax(corner6.z, corner7.z));
}

void getPrimitiveAABB(Geom& geom, glm::vec3& mins, glm::vec3& maxs)
{
    if (geom.type == TRIANGLE) {
        getTriangleAABB(geom, mins, maxs);
    }
    else {
        getOtherPrimitiveAABB(geom, mins, maxs);
    }
}

void Scene::expandBounds(int start, int end, glm::vec3& mins, glm::vec3& maxs)
{
    glm::vec3 otherMins;
    glm::vec3 otherMaxs;
    for (int i = start; i < end; ++i) {
        Geom& primitive = geoms[indices[i]];
        std::cout << " processing: " << indices[i];
        getPrimitiveAABB(primitive, otherMins, otherMaxs);
        std::cout << " mins: " << glm::to_string(otherMins) << " maxs: " << glm::to_string(otherMaxs) << std::endl;
        mins.x = fminf(mins.x, otherMins.x);
        mins.y = fminf(mins.y, otherMins.y);
        mins.z = fminf(mins.z, otherMins.z);
        maxs.x = fmaxf(maxs.x, otherMaxs.x);
        maxs.y = fmaxf(maxs.y, otherMaxs.y);
        maxs.z = fmaxf(maxs.z, otherMaxs.z);
    }
    std::cout << "processed mins: " << glm::to_string(mins) << " processed maxs: " << glm::to_string(maxs) << std::endl;
}

glm::vec3 getCentroid(Geom& geom)
{
    glm::vec3 mins;
    glm::vec3 maxs;
    getPrimitiveAABB(geom, mins, maxs);
    return (mins + maxs) * 0.5f;
}

int Scene::getSubtreeSize(int nodeIndex) {
    BVHNode& node = nodes[nodeIndex];
    if (node.isLeaf()) {
        return 1;
    }
    else {
        int leftSubtreeSize = getSubtreeSize(node.leftChild);
        int rightSubtreeSize = getSubtreeSize(node.leftChild + getSubtreeSize(node.leftChild));
        return 1 + leftSubtreeSize + rightSubtreeSize;
    }
}

void Scene::buildBVHTree(int start, int end)
{
    BVHNode& node = nodes[nodes.size() - 1];
    std::cout << "Current Node: " << nodes.size() - 1 << std::endl;
    expandBounds(start, end, node.mins, node.maxs);

    int numPrimitives = end - start;
    if (numPrimitives <= 2) //stop dividing the tree further down
    {
        node.startIndex = start;
        node.numPrimitives = numPrimitives;
        return;
    }

    // find split axis
    int axis = 0;
    glm::vec3 extent = node.maxs - node.mins;
    if (extent.y > extent.x && extent.y > extent.z)
    {
        axis = 1; //split along y axis
    }
    else if (extent.z > extent.x) {
        axis = 2; //split along z axis
    }
    
    // partition indices array such that everything before mid have
    // centroids less than or equal to the centroid at mid
    int mid = (start + end) / 2;
    std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
        [&](int a, int b) {
            glm::vec3 centroidA = getCentroid(geoms[a]);
            glm::vec3 centroidB = getCentroid(geoms[b]);
            return centroidA[axis] < centroidB[axis];
        });

    node.leftChild = nodes.size();
    BVHNode leftChild = BVHNode();
    nodes.push_back(leftChild);
    buildBVHTree(start, mid);

    node.rightChild = nodes.size();
    BVHNode rightChild = BVHNode();
    rightChild.startIndex = mid;
    nodes.push_back(rightChild);
    buildBVHTree(mid, end);
}

void Scene::constructBVHTree()
{
    indices.resize(geoms.size());
    std::iota(indices.begin(), indices.end(), 0);

    nodes.reserve(geoms.size() * 2 - 1);
    BVHNode root; 
    nodes.push_back(root); 

    buildBVHTree(0, geoms.size());
}

