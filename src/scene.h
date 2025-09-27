#pragma once

#include "sceneStructs.h"
#include <vector>
#include <numeric>

struct OctreeGPU {
    OctNode* d_nodes = nullptr;
    uint32_t* d_primIndex = nullptr;
    uint32_t   nodeCount = 0;
    uint32_t   primCount = 0;
    bool valid() const { return d_nodes && nodeCount > 0; }
};

class Scene
{
    struct OctBuildParams {
        int   maxDepth = 10;
        int   leafMax = 8;
        int   maxChildHits = 9;
        float loosen = 1.05f;
        float gainThresh = 1.6f;
    };

    struct BuildNode {
        AABB box;
        std::vector<uint32_t> primIds;
        int child[8]{ -1,-1,-1,-1,-1,-1,-1,-1 };
        bool leaf = false;
    };

private:
    void loadFromJSON(const std::string& jsonName);
    AABB worldAABBForGeom(const Geom& g);
    std::vector<PrimRef> buildPrimRefs(const std::vector<Geom>& geoms);
    void buildOctreeAndFlatten();
    bool overlap(const AABB& a, const AABB& b);
    void splitAABB8(const AABB& box, AABB childBox[8], float loosen);
    int buildNode(std::vector<BuildNode>& tmp, int nodeIdx, const std::vector<PrimRef>& prims,
        const OctBuildParams& P, int depth);
    void flattenRoot(const std::vector<Scene::BuildNode>& tmp,
        std::vector<OctNode>& outNodes,
        std::vector<uint32_t>& outPrimIndex);
    void flattenNodeAt(const std::vector<Scene::BuildNode>& tmp, int i,
        std::vector<OctNode>& outNodes,
        std::vector<uint32_t>& outPrimIndex,
        int outIndex);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<PrimRef> prims;
    std::vector<OctNode>   octNodes;
    std::vector<uint32_t>  octPrimIndex;

    OctreeGPU octGPU;

    void uploadOctreeToGPU();
    void freeOctreeGPU();
};
