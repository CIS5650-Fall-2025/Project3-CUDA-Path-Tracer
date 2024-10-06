#include "bvh.h"

#define MAX_DEPTH 32
#define NUM_SPLIT_TESTS 5

NodeList::NodeList(): nodes(nullptr), index(0), capacity(256) {
    nodes = new BVHNode[capacity];  // Initially allocate 256 nodes
}

NodeList::~NodeList() {
    delete[] nodes;
}

void NodeList::resizeArray() {
    int newCapacity = capacity * 2;  // Double the capacity
    BVHNode *newNodes = new BVHNode[newCapacity];  // Allocate new larger array

    // Copy old nodes to the new array
    std::memcpy(newNodes, nodes, capacity * sizeof(BVHNode));

    // Delete the old array
    delete[] nodes;

    // Update the pointer and capacity
    nodes = newNodes;
    capacity = newCapacity;
}

int NodeList::addNode(const BVHNode &node) {
    if (index >= capacity) {
        resizeArray();  // Resize if necessary
    }

    int nodeIndex = index;
    nodes[index] = node; 
    index++;

    return nodeIndex;
}

BVH::BVH() : allNodes(NodeList()), allBvhTris(nullptr), allTris(nullptr) {}

BVH::~BVH() {
    delete[] allBvhTris; 
    delete[] allTris;
}

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

void BVH::convertTrianglesToVertsIndicesNormals(const std::vector<Triangle>& triangles,
                                           std::vector<glm::vec3>& verts,
                                           std::vector<int>& indices,
                                           std::vector<glm::vec3>& normals) {

    std::unordered_map<glm::vec3, int, Vec3Hash, Vec3Equal> vertMap;  // Map to store unique vertices and their indices

    for (const auto& triangle : triangles) {
        // Process each vertex in the triangle
        for (int i = 0; i < 3; ++i) {
            const glm::vec3& vertex = triangle.points[i];
            const glm::vec3& normal = triangle.normals[i];

            // If the vertex is not already in the map, add it to verts and normals
            if (vertMap.find(vertex) == vertMap.end()) {
                vertMap[vertex] = verts.size();  // Add vertex to the map with current index
                verts.push_back(vertex);         // Add vertex to verts
                normals.push_back(normal);       // Add corresponding normal to normals
            }

            // Add the index of the vertex to the indices array
            indices.push_back(vertMap[vertex]);
        }
    }
}

void BVH::flattenTriangles(const std::vector<Triangle> &triangles, glm::vec3*& verts, int*& indices, glm::vec3*& normals, int &vertCount, int &indexCount) {
    // Populate verts, indices, and normals from the triangles
    std::vector<glm::vec3> vertsVec, normalsVec;
    std::vector<int> indicesVec;
    convertTrianglesToVertsIndicesNormals(triangles, vertsVec, indicesVec, normalsVec);

    // Set vertCount and indexCount for the output arrays
    vertCount = vertsVec.size();
    indexCount = indicesVec.size();

    // Allocate memory for output arrays
    verts = new glm::vec3[vertCount];    // Now correctly modifying verts
    normals = new glm::vec3[vertCount];  // Now correctly modifying normals
    indices = new int[indexCount];       // Now correctly modifying indices

    // Copy data from the vectors to the allocated arrays
    std::memcpy(verts, vertsVec.data(), vertCount * sizeof(glm::vec3));
    std::memcpy(normals, normalsVec.data(), vertCount * sizeof(glm::vec3));
    std::memcpy(indices, indicesVec.data(), indexCount * sizeof(int));
}

BVH::BVH(const std::vector<Triangle> &triangles) {
    glm::vec3* verts;
    glm::vec3* normals;
    int* indices;
    int vertCount, indexCount;
    flattenTriangles(triangles, verts, indices, normals, vertCount, indexCount);
    printf("Number of vertices/normals: %d\n", vertCount);
    printf("Number of indices: %d\n", indexCount);

    int numOfTriangles = triangles.size();
    allBvhTris = new BVHTriangle[numOfTriangles];
    BoundingBox bbox = BoundingBox();

    // Loop through all triangles, populate allBvhTris and calculate the size of the initial bounding box
    for (int i = 0; i < indexCount; i += 3) {
        glm::vec3 v1 = verts[indices[i]];
        glm::vec3 v2 = verts[indices[i + 1]];
        glm::vec3 v3 = verts[indices[i + 2]];

        glm::vec3 center = (v1 + v2 + v3) / 3.0f;
        glm::vec3 min = glm::min(v1, glm::min(v2, v3));
        glm::vec3 max = glm::max(v1, glm::max(v2, v3));
        
        allBvhTris[i / 3] = BVHTriangle(center, min, max, i / 3); // i / 3 is the index of the triangle
        bbox.resize(min, max);
    }
    
    printf("Initial bounding box size: (%f, %f, %f)\n", bbox.size.x, bbox.size.y, bbox.size.z);

    allNodes = NodeList();
    allNodes.addNode(BVHNode(bbox)); // Add the root node
    printf("numOfTriangles: %d\n", numOfTriangles);
    split(0, verts, 0, numOfTriangles);

    return;

    allTris = new Triangle[numOfTriangles];
    for (int i = 0; i < numOfTriangles; i++) {
        BVHTriangle buildTri = allBvhTris[i];

        glm::vec3 a = verts[indices[buildTri.index]];
        glm::vec3 b = verts[indices[buildTri.index + 1]];
        glm::vec3 c = verts[indices[buildTri.index + 2]];
        glm::vec3 norm_a = normals[indices[buildTri.index]];
        glm::vec3 norm_b = normals[indices[buildTri.index + 1]];
        glm::vec3 norm_c = normals[indices[buildTri.index + 2]];

        allTris[i] = Triangle(a, b, c, norm_a, norm_b, norm_c);
        Triangle tri = allTris[i];
        printf("Triangle %d: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", i, tri.points[0].x, tri.points[0].y, tri.points[0].z, tri.points[1].x, tri.points[1].y, tri.points[1].z, tri.points[2].x, tri.points[2].y, tri.points[2].z);
    }
}


float BVH::computeNodeCost(glm::vec3 size, int numTriangles) {
    float halfArea = size.x * size.y + size.x * size.z + size.y * size.z;
    return halfArea * numTriangles;
}

float BVH::evalSplit(int splitAxis, float splitPos, int start, int count) {
    BoundingBox boundsLeft = BoundingBox();
    BoundingBox boundsRight = BoundingBox();
    int numOnLeft = 0;
    int numOnRight = 0;

    for (int i = start; i < start + count; i++)
    {
        BVHTriangle tri = allBvhTris[i];
        if (tri.center[splitAxis] < splitPos)
        {
            boundsLeft.resize(tri.triMinCoors, tri.triMaxCoors);
            numOnLeft++;
        }
        else
        {
            boundsRight.resize(tri.triMinCoors, tri.triMaxCoors);
            numOnRight++;
        }
    }

    float costA = computeNodeCost(boundsLeft.size, numOnLeft);
    float costB = computeNodeCost(boundsRight.size, numOnRight);

    return costA + costB;
}

std::tuple<int, float, float> BVH::chooseSplit(BVHNode node, int start, int count) {
    if (count <= 1) {
        return std::make_tuple(0, 0.0f, std::numeric_limits<float>::infinity());
    }

    float bestSplitPos = 0.0f;
    int bestSplitAxis = 0;
    float bestCost = std::numeric_limits<float>::max();

    for (int axis = 0; axis < 3; axis++)
    {
        for (int i = 0; i < NUM_SPLIT_TESTS; i++)
        {
            float splitT = ((float)i + 1.0f) / (NUM_SPLIT_TESTS + 1.0f);
            float nMinAtAxis = node.nMinCoors[axis];
            float splitPos = nMinAtAxis + splitT * (node.nMaxCoors[axis] - nMinAtAxis);
            float cost = evalSplit(axis, splitPos, start, count);

            if (cost < bestCost)
            {
                bestCost = cost;
                bestSplitPos = splitPos;
                bestSplitAxis = axis;
            }
        }
    }

    return std::make_tuple(bestSplitAxis, bestSplitPos, bestCost);  
}

void BVH::split(int parentIndex, glm::vec3* verts, int triGlobalStart, int triNum, int depth) {
    BVHNode parent = allNodes.nodes[parentIndex];
    glm::vec3 size = parent.computeBboxSize();
    float parentCost = computeNodeCost(size, triNum);

    std::tuple<int, float, float> splitInfo = chooseSplit(parent, triGlobalStart, triNum);
    int splitAxis = std::get<0>(splitInfo);
    float splitPos = std::get<1>(splitInfo);
    float cost = std::get<2>(splitInfo);

    if (cost < parentCost && depth < MAX_DEPTH) {
        BoundingBox boundsLeft = BoundingBox();
        BoundingBox boundsRight = BoundingBox();
        int numOnLeft = 0;

        for (int i = triGlobalStart; i < triGlobalStart + triNum; i++)
        {
            // printf("Current iteration: %d\n", i);
            BVHTriangle tri = allBvhTris[i];
            if (tri.center[splitAxis] < splitPos)
            {
                boundsLeft.resize(tri.triMinCoors, tri.triMaxCoors);

                int startIdx = triGlobalStart + numOnLeft;

                BVHTriangle swap = allBvhTris[startIdx];
                printf("Swap triangle: %d\n", swap.index);
                allBvhTris[startIdx] = tri;
                printf("check swapped triangle: %d\n", allBvhTris[startIdx].index);
                allBvhTris[i] = swap;
                printf("check original triangle: %d\n", allBvhTris[i].index);
                numOnLeft++;                
            }
            else
            {
                boundsRight.resize(tri.triMinCoors, tri.triMaxCoors);
            }
        }

        int numOnRight = triNum - numOnLeft;
        int triStartLeft = triGlobalStart;
        int triStartRight = triGlobalStart + numOnLeft;

        // Split parent into two children
        int childIndexLeft = allNodes.addNode(BVHNode(boundsLeft, triStartLeft, 0));
        int childIndexRight = allNodes.addNode(BVHNode(boundsRight, triStartRight, 0));

        // Update parent
        if (parentIndex >= 0 && parentIndex < allNodes.capacity) {
            parent.startIdx = childIndexLeft;
            allNodes.nodes[parentIndex] = parent;
        } else {
            printf("Error: Invalid parentIndex in split()\n");
            return;
        }

        // Recursively split children
        if (numOnLeft > 0 && numOnRight > 0) {
            split(childIndexLeft, verts, triGlobalStart, numOnLeft, depth + 1);
            split(childIndexRight, verts, triGlobalStart + numOnLeft, numOnRight, depth + 1);
        } 
        else {
            printf("Error: Invalid split counts in split()\n");
            return;
        }

    }
    else
    {
        printf("Parent index: %d\n", parentIndex);
        // Parent is actually leaf, assign all triangles to it
        parent.startIdx = triGlobalStart;
        parent.numOfTris = triNum;
        allNodes.nodes[parentIndex] = parent;
    }
}