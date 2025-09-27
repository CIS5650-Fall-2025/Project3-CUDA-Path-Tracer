#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>

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
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

bool Scene::overlap(const AABB& a, const AABB& b) {
    return (a.bmin.x <= b.bmax.x && a.bmax.x >= b.bmin.x) &&
        (a.bmin.y <= b.bmax.y && a.bmax.y >= b.bmin.y) &&
        (a.bmin.z <= b.bmax.z && a.bmax.z >= b.bmin.z);
}

void Scene::splitAABB8(const AABB& box, AABB childBox[8], float loosen) {
    glm::vec3 c = 0.5f * (box.bmin + box.bmax);
    glm::vec3 e = 0.5f * (box.bmax - box.bmin) * loosen;
    glm::vec3 m = c - e, M = c + e;

    childBox[0] = { {m.x,m.y,m.z}, {c.x,c.y,c.z} };
    childBox[1] = { {c.x,m.y,m.z}, {M.x,c.y,c.z} };
    childBox[2] = { {m.x,c.y,m.z}, {c.x,M.y,c.z} };
    childBox[3] = { {c.x,c.y,m.z}, {M.x,M.y,c.z} };
    childBox[4] = { {m.x,m.y,c.z}, {c.x,c.y,M.z} };
    childBox[5] = { {c.x,m.y,c.z}, {M.x,c.y,M.z} };
    childBox[6] = { {m.x,c.y,c.z}, {c.x,M.y,M.z} };
    childBox[7] = { {c.x,c.y,c.z}, {M.x,M.y,M.z} };
}

int Scene::buildNode(std::vector<BuildNode>& tmp, int nodeIdx,
    const std::vector<PrimRef>& prims,
    const OctBuildParams& P, int depth)
{
    BuildNode& N = tmp[nodeIdx];
    if ((int)N.primIds.size() <= P.leafMax || depth >= P.maxDepth) {
        N.leaf = true; return nodeIdx;
    }

    AABB childBox[8]; splitAABB8(N.box, childBox, P.loosen);

    std::vector<uint32_t> childPrims[8], keep;
    keep.reserve(N.primIds.size());
    size_t sumChild = 0;

    for (uint32_t pid : N.primIds) {
        int hits = 0;
        bool h[8]{};
        for (int c = 0;c < 8;++c) { h[c] = overlap(prims[pid].aabb, childBox[c]); hits += (int)h[c]; }
        if (hits == 0 || hits >= P.maxChildHits) { keep.push_back(pid); continue; }
        for (int c = 0;c < 8;++c) if (h[c]) childPrims[c].push_back(pid);
        sumChild += (size_t)hits;
    }

    if (sumChild + keep.size() >= (size_t)(N.primIds.size() * P.gainThresh)) {
        N.leaf = true; return nodeIdx;
    }

    bool anyChild = false;
    for (int c = 0;c < 8;++c) anyChild |= !childPrims[c].empty();
    if (!anyChild) { N.leaf = true; return nodeIdx; }

    for (int c = 0;c < 8;++c) {
        if (childPrims[c].empty()) continue;
        N.child[c] = (int)tmp.size();
        tmp.push_back({ childBox[c] });
        tmp.back().primIds = std::move(childPrims[c]);
        buildNode(tmp, N.child[c], prims, P, depth + 1);
    }

    if (!keep.empty()) {
        N.leaf = true;
        N.child[0] = N.child[1] = N.child[2] = N.child[3] = N.child[4] = N.child[5] = N.child[6] = N.child[7] = -1;
        N.primIds.swap(keep);
    }
    else {
        N.primIds.clear();
    }
    return nodeIdx;
}

void Scene::flattenNodeAt(const std::vector<Scene::BuildNode>& tmp, int i,
    std::vector<OctNode>& outNodes,
    std::vector<uint32_t>& outPrimIndex,
    int outIndex)
{
    const Scene::BuildNode& N = tmp[i];
    OctNode& O = outNodes[outIndex];
    O = {};                       
    O.bmin = N.box.bmin;
    O.bmax = N.box.bmax;

    if (N.leaf) {
        O.childCount = 0;
        O.firstPrim = (uint32_t)outPrimIndex.size();
        O.primCount = (uint32_t)N.primIds.size();
        for (uint32_t pid : N.primIds) outPrimIndex.push_back(pid);
        return;
    }

    int childIds[8]; int k = 0;
    for (int c = 0;c < 8;++c) if (N.child[c] != -1) childIds[k++] = N.child[c];
    if (k == 0) {
        O.childCount = 0;
        O.firstPrim = (uint32_t)outPrimIndex.size();
        O.primCount = (uint32_t)N.primIds.size();
        for (uint32_t pid : N.primIds) outPrimIndex.push_back(pid);
        return;
    }

    O.firstChild = (uint32_t)outNodes.size();
    O.childCount = (uint16_t)k;
    size_t base = outNodes.size();
    outNodes.resize(base + k);         

    for (int j = 0;j < k;++j) {
        int childOutIndex = (int)base + j;
        flattenNodeAt(tmp, childIds[j], outNodes, outPrimIndex, childOutIndex);
    }
}

void Scene::flattenRoot(const std::vector<Scene::BuildNode>& tmp,
    std::vector<OctNode>& outNodes,
    std::vector<uint32_t>& outPrimIndex)
{
    outNodes.clear(); outPrimIndex.clear();
    outNodes.resize(1);                
    flattenNodeAt(tmp, 0, outNodes, outPrimIndex, 0);
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
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - roughness;
        }
        else if (p["TYPE"] == "Transmissive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const auto& eta = p["ETA"];
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = eta;
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
#if OCTREE
    this->prims = buildPrimRefs(geoms);
    this->buildOctreeAndFlatten();
#endif

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

    camera.apertureRadius = cameraData["APERTURERADIUS"];
    camera.focalDistance = glm::length(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

AABB Scene::worldAABBForGeom(const Geom& g) {
    if (g.type == SPHERE) {
        const glm::mat4& M = g.transform;
        glm::vec3 C = glm::vec3(M * glm::vec4(0, 0, 0, 1));
        float sx = glm::length(glm::vec3(M[0]));
        float sy = glm::length(glm::vec3(M[1]));
        float sz = glm::length(glm::vec3(M[2]));
        float r = 0.5f * fmaxf(sx, fmaxf(sy, sz)); 
        return { C - glm::vec3(r), C + glm::vec3(r) };
    }
    else {
        static const glm::vec3 corners[8] = {
            {-0.5f,-0.5f,-0.5f},{+0.5f,-0.5f,-0.5f},{-0.5f,+0.5f,-0.5f},{+0.5f,+0.5f,-0.5f},
            {-0.5f,-0.5f,+0.5f},{+0.5f,-0.5f,+0.5f},{-0.5f,+0.5f,+0.5f},{+0.5f,+0.5f,+0.5f}
        };
        const glm::mat4& M = g.transform;
        glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
        for (int i = 0;i < 8;++i) {
            glm::vec3 p = glm::vec3(M * glm::vec4(corners[i], 1.f));
            bmin = glm::min(bmin, p);
            bmax = glm::max(bmax, p);
        }
        return { bmin, bmax };
    }
}

std::vector<PrimRef> Scene::buildPrimRefs(const std::vector<Geom>& geoms) {
    std::vector<PrimRef> out;
    out.reserve(geoms.size());
    for (uint32_t i = 0; i < geoms.size(); ++i) {
        AABB a = worldAABBForGeom(geoms[i]);
        out.push_back(PrimRef{ i, a });
    }
    return out;
}

void Scene::buildOctreeAndFlatten()
{
    AABB scene = { glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX) };
    for (const auto& p : this->prims) {
        scene.bmin = glm::min(scene.bmin, p.aabb.bmin);
        scene.bmax = glm::max(scene.bmax, p.aabb.bmax);
    }

    std::vector<BuildNode> tmp; tmp.reserve(prims.size() * 2);
    tmp.push_back({ scene });
    tmp[0].primIds.resize(prims.size());
    std::iota(tmp[0].primIds.begin(), tmp[0].primIds.end(), 0u);

    OctBuildParams P;
    buildNode(tmp, 0, this->prims, P, 0);

    flattenRoot(tmp, this->octNodes, this->octPrimIndex);

     std::cout << "Octree nodes: " << octNodes.size()
               << ", leaf prims: " << octPrimIndex.size() << std::endl;
}

void Scene::freeOctreeGPU() {
    if (octGPU.d_nodes) { cudaFree(octGPU.d_nodes);     octGPU.d_nodes = nullptr; }
    if (octGPU.d_primIndex) { cudaFree(octGPU.d_primIndex); octGPU.d_primIndex = nullptr; }
    octGPU.nodeCount = 0;
    octGPU.primCount = 0;
}

void Scene::uploadOctreeToGPU() {
    freeOctreeGPU();

    if (octNodes.empty()) return;

    cudaMalloc(&octGPU.d_nodes, octNodes.size() * sizeof(OctNode));
    cudaMalloc(&octGPU.d_primIndex, octPrimIndex.size() * sizeof(uint32_t));

    cudaMemcpy(octGPU.d_nodes, octNodes.data(), octNodes.size() * sizeof(OctNode), cudaMemcpyHostToDevice);
    cudaMemcpy(octGPU.d_primIndex, octPrimIndex.data(), octPrimIndex.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    octGPU.nodeCount = static_cast<uint32_t>(octNodes.size());
    octGPU.primCount = static_cast<uint32_t>(octPrimIndex.size());
}