#pragma once

#include "glm/glm.hpp"
#include <numeric>
#include <algorithm>

#include "sceneStructs.h"


// BVH Construction inspiration:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
// https://jacco.ompf2.com/2022/04/18/how-to-build-a-bvh-part-2-faster-rays/

////////////////////////////////// AABB Struct //////////////////////////////////
struct AABB {
    glm::vec3 min = glm::vec3(FLT_MAX);         // Minimum corner of the bounding box
    glm::vec3 max = glm::vec3(-FLT_MAX);        // Maximum corner of the bounding box
    void expand(const AABB& other) {            // Expands the AABB to include another AABB
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }
    void expand(const const glm::vec3& point) { // Expands the AABB to include a point
        min = glm::min(min, point);
        max = glm::max(max, point);
    }
    float getSurfaceArea() {                    // Get surface area of an AABB
        glm::vec3 extent = max - min;
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }
    void getSplitPlaneAxisAndPosition_ByLongestAxis(int& splitAxis, float& splitPosition) {
        glm::vec3 extent = max - min;
        splitAxis = 0;
        if (extent.y > extent.x) splitAxis = 1;
        if (extent.z > extent[splitAxis]) splitAxis = 2;
        splitPosition = min[splitAxis] + extent[splitAxis] * 0.5f;
    }
};

// General AABB calculation, calls one of below methods based on Geom.type
AABB calculateAABB(const Geom& geom);
// Functions to calculate AABBs for different geometries
AABB calculateAABBTriangle(const Geom& triangle);
AABB calculateAABBCube(const Geom& cube);
AABB calculateAABBSphere(const Geom& sphere);


////////////////////////////////// BVHNode Struct //////////////////////////////////
struct BVHNode {
    AABB aabb;
    int leftFirst;
    int geomCount;
    bool isLeaf() { return geomCount > 0; }
};

// General centroid calculation, calls one of below methods based on Geom.type
glm::vec3 calculateCentroid(const Geom& geom);
// Functions to calculate centroids for different geometries
glm::vec3 calculateCentroidTriangle(const Geom& triangle);
glm::vec3 calculateCentroidCube(const Geom& cube);
glm::vec3 calculateCentroidSphere(const Geom& sphere);

////////////////////////////////// BVHBuilder Class //////////////////////////////////
class BVHBuilder
{
private:
    std::vector<Geom>* geoms = nullptr;
    std::vector<int> geomIdx;
    std::vector<BVHNode>* bvhNodes = nullptr;
    int rootNodeIdx;
    int nodesUsed;

    void initBVH();
    void UpdateNodeBounds(int nodeIdx);
    void Subdivide(int nodeIdx);
    int getSplitPlaneAxisAndPosition_SurfaceAreaHueristic(int& splitAxis, float& splitPosition, const BVHNode& node);

public:
    BVHBuilder(std::vector<Geom>& _geoms, std::vector<BVHNode>& _bvhNodes);
    ~BVHBuilder();

    int buildBVH(std::vector<int>& _geomIdx);
};