#include "bvh.h"

#define MAX_DEPTH 32
#define NUM_SPLIT_TESTS 5

NodeList::NodeList(): index(0), capacity(256) {
    nodes = new BVHNode[capacity];
}

NodeList::~NodeList() {}

void NodeList::resize(int newCapacity) {
    BVHNode* newNodes = new BVHNode[newCapacity];   // Allocate new array with larger capacity

    if (index > newCapacity) {
        printf("Your current number of nodes is: %d, which is larger than your NodeList capacity: %d\n", index, newCapacity);
        return;
    }

    // Copy existing nodes to the new array
    for (int i = 0; i < index; ++i) {
        newNodes[i] = nodes[i];
    }

    nodes = newNodes;  // Point to the new array
    capacity = newCapacity;  // Update the capacity
}

int NodeList::addNode(const BVHNode& node) {
    if (index >= capacity) {
        resize(capacity * 2);  // Double the capacity if array is full
    }

    int nodeIndex = index;
    nodes[index] = node;
    index++;
    return nodeIndex;
}

const int NodeList::nodeCount() {
    return index;
}

BVH::BVH() : allNodes(NodeList()), allBvhTriangles(nullptr), allTriangles(nullptr) {}

BVH::BVH(const std::vector<Triangle> &triangles) {
    glm::vec3* verts;
    glm::vec3* normals;
    int* indices;
    int vertCount, indexCount;
    // Calling this function ensures that we have the same number of vertices and normals
    flattenTriangles(triangles, verts, indices, normals, vertCount, indexCount); 

    allNodes = NodeList();
    int numTris = indexCount / 3;
    // Dynamic allocation for triangle data (manual memory management)
    allBvhTriangles = new BVHTriangle[numTris];
    BoundingBox bounds = BoundingBox();

    for (int i = 0; i < indexCount; i += 3) {
        glm::vec3 a = verts[indices[i + 0]];
        glm::vec3 b = verts[indices[i + 1]];
        glm::vec3 c = verts[indices[i + 2]];
        glm::vec3 centre = (a + b + c) / 3.0f;

        glm::vec3 max = glm::max(glm::max(a, b), c);
        glm::vec3 min = glm::min(glm::min(a, b), c);

        int curIdx = i / 3;
        if (curIdx > numTris || curIdx < 0) {
            printf("You are assining your BVHTriangle at an illegal position.");
            return;
        }
        allBvhTriangles[curIdx] = BVHTriangle(centre, min, max, i);
        bounds.resize(min, max);
    }

    allNodes.addNode(BVHNode(bounds));
    splitNode(0, verts, 0, numTris);

    //allTris = new Triangle[AllTriangles.Length];
    allTriangles = new Triangle[numTris];
    printf("Number of triangles in BVH: %d\n", numTris);
    // Creating final triangles for rendering or further processing
    for (int i = 0; i < numTris; i++) {
        BVHTriangle buildTri = allBvhTriangles[i];
        glm::vec3 a = verts[indices[buildTri.index + 0]];
        glm::vec3 b = verts[indices[buildTri.index + 1]];
        glm::vec3 c = verts[indices[buildTri.index + 2]];
        glm::vec3 norm_a = normals[indices[buildTri.index + 0]];
        glm::vec3 norm_b = normals[indices[buildTri.index + 1]];
        glm::vec3 norm_c = normals[indices[buildTri.index + 2]];

        allTriangles[i] = Triangle(a, b, c, norm_a, norm_b, norm_c);
    }
}

BVH::~BVH() {
    // Manual memory management cleanup
    int numNodes = allNodes.nodeCount();
    if (numNodes > 0) {
        delete[] allNodes.nodes;
    }
    delete[] allBvhTriangles;
    delete[] allTriangles;
}

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

float BVH::lerp(float a, float b, float t) {
        return a + t * (b - a);
    }

float BVH::computeNodeCost(const glm::vec3& size, int numTriangles) {
    float halfArea = size.x * size.y + size.x * size.z + size.y * size.z;
    return halfArea * numTriangles;
}

float BVH::evalSplit(int splitAxis, float splitPos, int start, int count) {
    BoundingBox boundsLeft = BoundingBox();
    BoundingBox boundsRight = BoundingBox();
    int numOnLeft = 0;
    int numOnRight = 0;

    // Loop through all triangles in the range [start, start + count)
    for (int i = start; i < start + count; ++i) {
        BVHTriangle& tri = allBvhTriangles[i];

        // Split based on the axis and position
        if (tri.center[splitAxis] < splitPos) {
            boundsLeft.resize(tri.minCoors, tri.maxCoors);
            numOnLeft++;
        }
        else {
            boundsRight.resize(tri.minCoors, tri.maxCoors);
            numOnRight++;
        }
    }

    // Calculate the cost for both sides of the split
    float costA = computeNodeCost(boundsLeft.Size(), numOnLeft);
    float costB = computeNodeCost(boundsRight.Size(), numOnRight);
    return costA + costB;
}

SplitResult BVH::chooseSplit(const BVHNode& node, int start, int count) {
    if (count <= 1) {
        return SplitResult(0, 0, std::numeric_limits<float>::infinity());
    }

    float bestSplitPos = 0;
    int bestSplitAxis = 0;
    float bestCost = std::numeric_limits<float>::max();

    // Estimate best split position
    for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < NUM_SPLIT_TESTS; ++i) {
            float splitT = (i + 1) / static_cast<float>(NUM_SPLIT_TESTS + 1);
            float splitPos = lerp(node.minCoors[axis], node.maxCoors[axis], splitT);

            float cost = evalSplit(axis, splitPos, start, count);
            if (cost < bestCost) {
                bestCost = cost;
                bestSplitPos = splitPos;
                bestSplitAxis = axis;
            }
        }
    }

    return SplitResult(bestSplitAxis, bestSplitPos, bestCost);
}

void BVH::splitNode(int parentIndex, const glm::vec3* verts, int triGlobalStart, int triNum, int depth) {
    BVHNode& parent = allNodes.nodes[parentIndex];
    glm::vec3 size = parent.calculateBoundsSize();
    float parentCost = computeNodeCost(size, triNum);

    // Get the split result (axis, position, and cost)
    SplitResult split = chooseSplit(parent, triGlobalStart, triNum);

    if (split.cost < parentCost && depth < MAX_DEPTH) {
        BoundingBox boundsLeft = BoundingBox();
        BoundingBox boundsRight = BoundingBox();
        int numOnLeft = 0;

        // Partition triangles into left and right nodes
        for (int i = triGlobalStart; i < triGlobalStart + triNum; ++i) {
            BVHTriangle& tri = allBvhTriangles[i];
            if (tri.center[split.axis] < split.pos) {
                boundsLeft.resize(tri.minCoors, tri.maxCoors);

                // Swap triangles to group left-side ones at the start of the array
                BVHTriangle temp = allBvhTriangles[triGlobalStart + numOnLeft];
                allBvhTriangles[triGlobalStart + numOnLeft] = tri;
                allBvhTriangles[i] = temp;
                numOnLeft++;
            }
            else {
                boundsRight.resize(tri.minCoors, tri.maxCoors);
            }
        }

        int numOnRight = triNum - numOnLeft;
        int triStartLeft = triGlobalStart;
        int triStartRight = triGlobalStart + numOnLeft;

        // Split parent into two children
        int childIndexLeft = allNodes.addNode(BVHNode(boundsLeft, triStartLeft, 0));
        int childIndexRight = allNodes.addNode(BVHNode(boundsRight, triStartRight, 0));

        // Update the parent node
        parent.startIdx = childIndexLeft;
        allNodes.nodes[parentIndex] = parent;

        // Recursively split the children
        splitNode(childIndexLeft, verts, triGlobalStart, numOnLeft, depth + 1);
        splitNode(childIndexRight, verts, triGlobalStart + numOnLeft, numOnRight, depth + 1);
    }
    else {
        // The node becomes a leaf; assign all triangles to it
        parent.startIdx = triGlobalStart;
        parent.numOfTriangles = triNum;
        allNodes.nodes[parentIndex] = parent;
    }
}
