#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>   // for FLT_MAX

#include "glm/glm.hpp"
#include "sceneStructs.h" 

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct BVHNode {
    AABB bound;
    int left;
    int right;
    int start;
    int end;
    bool isLeaf;
};

//A bin is simply a bucket or container that groups triangles based on their centroid position along the current splitting axis.
struct Bin {
    AABB bound;
    int count = 0;

    Bin() {
        bound.min = glm::vec3(FLT_MAX);
        bound.max = glm::vec3(-FLT_MAX);
        count = 0;
    }
};


struct SceneObject {
    int objectID;  // optional: for reference
    AABB bound;    // object's world-space AABB
    int bvhRootIndex; // root node index of its BLAS
};



AABB computeAABB(const Triangle& tri);
int constructBVH_MidpointSplit(std::vector<BVHNode>& nodes, std::vector<Triangle>& triangles, int start, int end);
int constructBVH_SAH(std::vector<BVHNode>& nodes, std::vector<Triangle>& triangles, int start, int end);
int constructBVH_SAH_Binned(std::vector<BVHNode>& nodes, std::vector<Triangle>& triangles, int start, int end);
//int constructTLAS_BinnedSAH(std::vector<BVHNode>& nodes, std::vector<SceneObject>& sceneObjects, int start, int end);