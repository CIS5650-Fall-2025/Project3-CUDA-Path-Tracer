#pragma once

#include <cstring>  // For memcpy
#include <tuple>
#include <limits>
#include <unordered_map>

#include "glm/glm.hpp"

#include "sceneStructs.h"

class NodeList {
private:
    int index;        // Current index

    // Function to resize the array when it's full
    void resizeArray();

public:
    BVHNode *nodes;   // Array of BVHNode objects
    int capacity;     // Current capacity of the array

    NodeList();
    ~NodeList();

    // Function to add a node, resizing if necessary
    int addNode(const BVHNode &node);
};

class BVH {
private:
    // Function to convert a vector of Triangle to verts, indices, and normals
    void convertTrianglesToVertsIndicesNormals(const std::vector<Triangle>& triangles,
                                           std::vector<glm::vec3>& verts,
                                           std::vector<int>& indices,
                                           std::vector<glm::vec3>& normals);
    
    // Function to flatten the triangles into verts, indices, and normals
    void flattenTriangles(const std::vector<Triangle> &triangles, glm::vec3*& verts, int*& indices, glm::vec3*& normals, int &vertCount, int &indexCount);

public:
    NodeList allNodes;
    BVHTriangle* allBvhTris;
    Triangle* allTris;

    BVH();
    BVH(const std::vector<Triangle> &triangles);
    ~BVH();

    BVHTriangle* getBVHTriangles() { return allBvhTris; }

    void split(int parentIndex, glm::vec3* verts, int triGlobalStart, int triNum, int depth = 0);
    std::tuple<int, float, float> chooseSplit(BVHNode node, int start, int count);
    float evalSplit(int splitAxis, float splitPos, int start, int count);
    float computeNodeCost(glm::vec3 size, int numTriangles);
};
