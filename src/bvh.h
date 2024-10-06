#pragma once

#include <cstring>  // For memcpy
#include <tuple>
#include <limits>
#include <unordered_map>

#include "glm/glm.hpp"

#include "sceneStructs.h"

class NodeList {
private:
    // Resize the Nodes array when the capacity is exceeded
    void resize(int newCapacity);

public:
    BVHNode* nodes;      // Pointer to dynamic array of Nodes
    int index;        // Current index to track number of nodes
    int capacity;     // Current capacity of the Nodes array

    // Constructor initializes array with initial capacity of 256 nodes
    NodeList();
    ~NodeList();

    // Add a Node to the list, resize if necessary
    int addNode(const BVHNode& node);

    // Return the current number of nodes
    const int nodeCount();
};

class BVH {
private:
    NodeList allNodes;
    BVHTriangle* allBvhTriangles;

    // Hash function for glm::vec3 to use with unordered_map
    struct Vec3Hash {
        std::size_t operator()(const glm::vec3& v) const {
            return std::hash<float>()(v.x) ^ std::hash<float>()(v.y) ^ std::hash<float>()(v.z);
        }
    };

    // Equality function for glm::vec3 to use with unordered_map
    struct Vec3Equal {
        bool operator()(const glm::vec3& v1, const glm::vec3& v2) const {
            return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
        }
    };

    float lerp(float a, float b, float t);

    // Function to convert a vector of Triangle to verts, indices, and normals
    void convertTrianglesToVertsIndicesNormals(const std::vector<Triangle>& triangles,
                                           std::vector<glm::vec3>& verts,
                                           std::vector<int>& indices,
                                           std::vector<glm::vec3>& normals);
    
    // Function to flatten the triangles into verts, indices, and normals
    void flattenTriangles(const std::vector<Triangle> &triangles, glm::vec3*& verts, int*& indices, glm::vec3*& normals, int &vertCount, int &indexCount);

    float computeNodeCost(const glm::vec3& size, int numTriangles);
    float evalSplit(int splitAxis, float splitPos, int start, int count);
    SplitResult chooseSplit(const BVHNode& node, int start, int count);
    void splitNode(int parentIndex, const glm::vec3* verts, int triGlobalStart, int triNum, int depth = 0);

public:
    Triangle* allTriangles;

    BVH();
    BVH(const std::vector<Triangle> &triangles);
    ~BVH();
};
