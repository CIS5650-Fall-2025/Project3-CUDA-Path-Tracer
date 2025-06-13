#include "BVH.h"
#include "sceneStructs.h"
#include <cfloat>  // for FLT_MAX


// calculate the minimum and maximum corner of the bounding box that fully contains this triangle.
AABB computeAABB(const Triangle& tri) {
    AABB box;
    box.min = glm::min(glm::min(tri.v0, tri.v1), tri.v2);
    box.max = glm::max(glm::max(tri.v0, tri.v1), tri.v2);
    return box;
}




int constructBVH_MidpointSplit(std::vector<BVHNode>& bvhNodes, std::vector<Triangle>& triangles, int startIndex, int endIndex) {
    // Create a new BVH node for the current range
    BVHNode currentNode;
    currentNode.start = startIndex;
    currentNode.end = endIndex;

    // Compute the bounding box for the current triangle range
    AABB boundingBox;
    boundingBox.min = glm::vec3(FLT_MAX);
    boundingBox.max = glm::vec3(-FLT_MAX);

    for (int i = startIndex; i < endIndex; i++) {
        AABB triangleAABB = computeAABB(triangles[i]);
        boundingBox.min = glm::min(boundingBox.min, triangleAABB.min);
        boundingBox.max = glm::max(boundingBox.max, triangleAABB.max);
    }
    currentNode.bound = boundingBox;

    // Calculate the number of triangles in this node
    int triangleCount = endIndex - startIndex;

    // If this is a leaf node (0 or 1 triangle), stop recursion
    if (triangleCount <= 1) {
        currentNode.isLeaf = true;
        currentNode.left = -1;
        currentNode.right = -1;

        bvhNodes.push_back(currentNode);
        return static_cast<int>(bvhNodes.size()) - 1;
    }

    // Internal node: determine split
    currentNode.isLeaf = false;

    glm::vec3 boxExtent = boundingBox.max - boundingBox.min;

    // Select split axis (longest axis)
    int splitAxis = (boxExtent.x > boxExtent.y) ? ((boxExtent.x > boxExtent.z) ? 0 : 2)
        : ((boxExtent.y > boxExtent.z) ? 1 : 2);

    // Helper lambda to compute centroid along split axis
    auto computeCentroid = [splitAxis](const Triangle& tri) {
        return (tri.v0[splitAxis] + tri.v1[splitAxis] + tri.v2[splitAxis]) / 3.0f;
        };

    // Sort triangles based on centroid along split axis
    std::sort(triangles.begin() + startIndex, triangles.begin() + endIndex,
        [&computeCentroid](const Triangle& a, const Triangle& b) {
            return computeCentroid(a) < computeCentroid(b);
        }
    );

    // Split triangles into two groups at midpoint
    int midIndex = startIndex + triangleCount / 2;

    // Recursively build left and right subtrees
    int leftChildIndex = constructBVH_MidpointSplit(bvhNodes, triangles, startIndex, midIndex);
    int rightChildIndex = constructBVH_MidpointSplit(bvhNodes, triangles, midIndex, endIndex);

    // Assign child indices
    currentNode.left = leftChildIndex;
    currentNode.right = rightChildIndex;

    // Insert current node into BVH
    bvhNodes.push_back(currentNode);
    return static_cast<int>(bvhNodes.size()) - 1;
}



// Compute surface area of an AABB
float surfaceArea(const AABB& box) {
    glm::vec3 d = box.max - box.min;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}


int constructBVH_SAH(std::vector<BVHNode>& bvhNodes, std::vector<Triangle>& triangles, int startIndex, int endIndex) {
    BVHNode currentNode;
    currentNode.start = startIndex;
    currentNode.end = endIndex;

    // Compute bounding box for current range
    AABB boundingBox;
    boundingBox.min = glm::vec3(FLT_MAX);
    boundingBox.max = glm::vec3(-FLT_MAX);
    for (int i = startIndex; i < endIndex; i++) {
        AABB triAABB = computeAABB(triangles[i]);
        boundingBox.min = glm::min(boundingBox.min, triAABB.min);
        boundingBox.max = glm::max(boundingBox.max, triAABB.max);
    }
    currentNode.bound = boundingBox;

    int triangleCount = endIndex - startIndex;

    if (triangleCount <= 1) {
        currentNode.isLeaf = true;
        currentNode.left = -1;
        currentNode.right = -1;
        bvhNodes.push_back(currentNode);
        return static_cast<int>(bvhNodes.size()) - 1;
    }

    // SAH split search
    const int numCandidates = 16;
    float bestCost = FLT_MAX;
    int bestAxis = -1;
    int bestSplit = -1;
    float parentArea = surfaceArea(boundingBox);

    for (int axis = 0; axis < 3; axis++) {
        auto computeCentroid = [axis](const Triangle& tri) {
            return (tri.v0[axis] + tri.v1[axis] + tri.v2[axis]) / 3.0f;
            };

        std::sort(triangles.begin() + startIndex, triangles.begin() + endIndex,
            [&computeCentroid](const Triangle& a, const Triangle& b) {
                return computeCentroid(a) < computeCentroid(b);
            });

        // Sweep all possible split positions
        for (int i = 1; i < triangleCount; i++) {
            AABB leftBox, rightBox;
            leftBox.min = glm::vec3(FLT_MAX);
            leftBox.max = glm::vec3(-FLT_MAX);
            rightBox.min = glm::vec3(FLT_MAX);
            rightBox.max = glm::vec3(-FLT_MAX);

            // Left AABB
            for (int j = startIndex; j < startIndex + i; j++) {
                AABB triAABB = computeAABB(triangles[j]);
                leftBox.min = glm::min(leftBox.min, triAABB.min);
                leftBox.max = glm::max(leftBox.max, triAABB.max);
            }
            // Right AABB
            for (int j = startIndex + i; j < endIndex; j++) {
                AABB triAABB = computeAABB(triangles[j]);
                rightBox.min = glm::min(rightBox.min, triAABB.min);
                rightBox.max = glm::max(rightBox.max, triAABB.max);
            }

            float leftArea = surfaceArea(leftBox);
            float rightArea = surfaceArea(rightBox);
            int leftCount = i;
            int rightCount = triangleCount - i;

            float sahCost = (leftArea / parentArea) * leftCount + (rightArea / parentArea) * rightCount;

            if (sahCost < bestCost) {
                bestCost = sahCost;
                bestAxis = axis;
                bestSplit = i;
            }
        }
    }

    // If SAH fails (very rare corner case), fallback to midpoint
    if (bestAxis == -1) {
        currentNode.isLeaf = true;
        currentNode.left = -1;
        currentNode.right = -1;
        bvhNodes.push_back(currentNode);
        return static_cast<int>(bvhNodes.size()) - 1;
    }

    // Perform best found split
    //During the SAH search, we loop over all 3 axes (axis = 0, 1, 2).
    //After finding the best axis, we must re - sort triangles only by the best axis for the final split.
    auto computeCentroid = [bestAxis](const Triangle& tri) {
        return (tri.v0[bestAxis] + tri.v1[bestAxis] + tri.v2[bestAxis]) / 3.0f;
        };

    std::sort(triangles.begin() + startIndex, triangles.begin() + endIndex,
        [&computeCentroid](const Triangle& a, const Triangle& b) {
            return computeCentroid(a) < computeCentroid(b);
        });

    int midIndex = startIndex + bestSplit;

    currentNode.isLeaf = false;
    int leftChild = constructBVH_SAH(bvhNodes, triangles, startIndex, midIndex);
    int rightChild = constructBVH_SAH(bvhNodes, triangles, midIndex, endIndex);

    currentNode.left = leftChild;
    currentNode.right = rightChild;

    bvhNodes.push_back(currentNode);
    return static_cast<int>(bvhNodes.size()) - 1;
}


// instead of testing every triangle individually (which is slow), we group triangles into bins so we only need to test a small number of split locations.
int constructBVH_SAH_Binned(std::vector<BVHNode>& bvhNodes, std::vector<Triangle>& triangles, int startIndex, int endIndex) {
    constexpr int NumBins = 16;

    BVHNode currentNode;
    currentNode.start = startIndex;
    currentNode.end = endIndex;

    // Compute bounding box for current range
    AABB boundingBox;
    boundingBox.min = glm::vec3(FLT_MAX);
    boundingBox.max = glm::vec3(-FLT_MAX);
    for (int i = startIndex; i < endIndex; i++) {
        AABB triAABB = computeAABB(triangles[i]);
        boundingBox.min = glm::min(boundingBox.min, triAABB.min);
        boundingBox.max = glm::max(boundingBox.max, triAABB.max);
    }
    currentNode.bound = boundingBox;

    int triangleCount = endIndex - startIndex;

    if (triangleCount <= 1) {
        currentNode.isLeaf = true;
        currentNode.left = -1;
        currentNode.right = -1;
        bvhNodes.push_back(currentNode);
        return static_cast<int>(bvhNodes.size()) - 1;
    }

    // Select longest axis
    glm::vec3 extent = boundingBox.max - boundingBox.min;
    int splitAxis = (extent.x > extent.y) ? ((extent.x > extent.z) ? 0 : 2)
        : ((extent.y > extent.z) ? 1 : 2);

    // Initialize bins
    Bin bins[NumBins];
    float binMin = boundingBox.min[splitAxis];
    float binMax = boundingBox.max[splitAxis];
    float binSize = (binMax - binMin) / NumBins;

    // Handle degenerate case
    if (binSize <= 0.0f) {
        currentNode.isLeaf = true;
        currentNode.left = -1;
        currentNode.right = -1;
        bvhNodes.push_back(currentNode);
        return static_cast<int>(bvhNodes.size()) - 1;
    }

    // Bin triangles
    auto computeCentroid = [splitAxis](const Triangle& tri) {
        return (tri.v0[splitAxis] + tri.v1[splitAxis] + tri.v2[splitAxis]) / 3.0f;
        };

    for (int i = startIndex; i < endIndex; i++) {
        float centroid = computeCentroid(triangles[i]);
        int binIdx = std::min(NumBins - 1, int((centroid - binMin) / binSize));
        bins[binIdx].bound.min = glm::min(bins[binIdx].bound.min, computeAABB(triangles[i]).min);
        bins[binIdx].bound.max = glm::max(bins[binIdx].bound.max, computeAABB(triangles[i]).max);
        bins[binIdx].count++;
    }

    // Precompute prefix sums of bounds and counts
    AABB leftBounds[NumBins];
    int leftCounts[NumBins];
    AABB rightBounds[NumBins];
    int rightCounts[NumBins];

    AABB leftBound = {};
    leftBound.min = glm::vec3(FLT_MAX);
    leftBound.max = glm::vec3(-FLT_MAX);
    int leftCount = 0;

    for (int i = 0; i < NumBins; i++) {
        leftBound.min = glm::min(leftBound.min, bins[i].bound.min);
        leftBound.max = glm::max(leftBound.max, bins[i].bound.max);
        leftCount += bins[i].count;
        leftBounds[i] = leftBound;
        leftCounts[i] = leftCount;
    }

    AABB rightBound = {};
    rightBound.min = glm::vec3(FLT_MAX);
    rightBound.max = glm::vec3(-FLT_MAX);
    int rightCount = 0;

    for (int i = NumBins - 1; i >= 0; i--) {
        rightBound.min = glm::min(rightBound.min, bins[i].bound.min);
        rightBound.max = glm::max(rightBound.max, bins[i].bound.max);
        rightCount += bins[i].count;
        rightBounds[i] = rightBound;
        rightCounts[i] = rightCount;
    }

    // Evaluate SAH cost for each bin boundary
    float parentArea = surfaceArea(boundingBox);
    float bestCost = FLT_MAX;
    int bestSplit = -1;

    for (int i = 0; i < NumBins - 1; i++) {
        float leftArea = surfaceArea(leftBounds[i]);
        float rightArea = surfaceArea(rightBounds[i + 1]);

        float cost = (leftArea / parentArea) * leftCounts[i] + (rightArea / parentArea) * rightCounts[i + 1];
        if (cost < bestCost) {
            bestCost = cost;
            bestSplit = i;
        }
    }

    // If no valid split found (degenerate), make leaf
    if (bestSplit == -1) {
        currentNode.isLeaf = true;
        currentNode.left = -1;
        currentNode.right = -1;
        bvhNodes.push_back(currentNode);
        return static_cast<int>(bvhNodes.size()) - 1;
    }

    // Partition triangles into left and right sets
    auto binAssignment = [splitAxis, binMin, binSize, NumBins](const Triangle& tri) {
        float centroid = (tri.v0[splitAxis] + tri.v1[splitAxis] + tri.v2[splitAxis]) / 3.0f;
        return std::min(int((centroid - binMin) / binSize), NumBins - 1);
        };


    auto midIter = std::partition(triangles.begin() + startIndex, triangles.begin() + endIndex,
        [&](const Triangle& tri) {
            return binAssignment(tri) <= bestSplit;
        });

    int midIndex = int(midIter - triangles.begin());

    // Recurse
    currentNode.isLeaf = false;
    int leftChild = constructBVH_SAH_Binned(bvhNodes, triangles, startIndex, midIndex);
    int rightChild = constructBVH_SAH_Binned(bvhNodes, triangles, midIndex, endIndex);

    currentNode.left = leftChild;
    currentNode.right = rightChild;
    bvhNodes.push_back(currentNode);
    return static_cast<int>(bvhNodes.size()) - 1;
}
