#pragma once

#include "bvh.h"
#include "sceneStructs.h"

using namespace std;

struct bbox;
struct bvhNode;

struct binPerAxis {
    bbox bbox;
    std::vector<int> indices; // indices of primitives in this bin
    float surfaceArea;
};

void Scene::buildBVH()
{   
    bvhNode root = bvhNode();
    // assume vertices and faces are already loaded
    for (size_t i = 0; i < mesh.faceIndices.size(); i++) {
        bbox this_bbox(mesh.vertices[mesh.faceIndices[i].x], 
                        mesh.vertices[mesh.faceIndices[i].y], 
                        mesh.vertices[mesh.faceIndices[i].z]);
        triangleBboxes.push_back(this_bbox);
        root.bbox.encloseBbox(this_bbox);
        mesh.faceIndicesBVH.push_back(i);
    }
    // construct root
    root.startIndex = 0;
    root.size = triangleBboxes.size();
    root.is_leaf = false;
    bvhNodes.push_back(root);

    buildBVHRecursive(bvhNodes[0], 0, bvhNodes[0].size);
    validateBVH();
}

void Scene::buildBVHRecursive(bvhNode& parent, int startIndex, int size) {
    /*
        1. check if parent should be leaf by compare size with max_leaf_size
        2. if not, find the best split axis by comparing SAH
        3. if SAH failed, make parent a leaf
            fail condition is best split result in all primitives in one bin ( can't have valid l and r )
        4. else, partition faceIndices, create two children, and assign them as the left and right of parent
    */

    // prepare variables
    std::vector<int>& faceIndicesBVH = mesh.faceIndicesBVH;

    // 1. check if parent should be leaf by compare size with max_leaf_size
    if (size <= max_leaf_size) {
        parent.setAsLeaf(startIndex, size);
        return;
    }

    // 2. find the best split axis by comparing SAH
    float min_cost = std::numeric_limits<float>::max();
    bool hasValidSplit = false;
    float parentSurfaceArea = parent.bbox.getSurfaceArea();

    int best_axis = -1;
    int best_split = -1;
    binPerAxis bestLeftBin, bestRightBin;

    for (int axis = 0; axis < 3; axis++) {
        std::vector<binPerAxis> thisAxisBins(binsToSplit);
        float binSpanMin = parent.bbox.min[axis];
        float binSpanMax = parent.bbox.max[axis];
        float binLength = (binSpanMax - binSpanMin) / binsToSplit; // length of each bin

        // compute bins for this axis
        for (int p = startIndex; p < startIndex + size; p++) {
            bbox& this_bbox = triangleBboxes[faceIndicesBVH[p]];
            float center = this_bbox.getCenter()[axis];
            int bin_idx = floor((center - binSpanMin) / binLength);
            thisAxisBins[bin_idx].bbox.encloseBbox(this_bbox);
            thisAxisBins[bin_idx].indices.push_back(faceIndicesBVH[p]);
            thisAxisBins[bin_idx].surfaceArea += this_bbox.getSurfaceArea();
        }
        // compute each split's cost, check if there's a valid split
        for (int split_start = 1; split_start < binsToSplit - 1; split_start++) {
            binPerAxis leftSplitBin, rightSplitBin;
            for (int split_left = 0; split_left < split_start; split_left++) {
                if (thisAxisBins[split_left].indices.size() == 0) continue; // empty split
                leftSplitBin.bbox.encloseBbox(thisAxisBins[split_left].bbox);
                leftSplitBin.surfaceArea += thisAxisBins[split_left].surfaceArea;
                leftSplitBin.indices.insert(leftSplitBin.indices.end(), 
                                            thisAxisBins[split_left].indices.begin(), 
                                            thisAxisBins[split_left].indices.end());
            }
            for (int split_right = split_start; split_right < binsToSplit; split_right++) {
                if (thisAxisBins[split_right].indices.size() == 0) continue; // empty split
                rightSplitBin.bbox.encloseBbox(thisAxisBins[split_right].bbox);
                rightSplitBin.surfaceArea += thisAxisBins[split_right].surfaceArea;
                rightSplitBin.indices.insert(rightSplitBin.indices.end(), 
                                            thisAxisBins[split_right].indices.begin(), 
                                            thisAxisBins[split_right].indices.end());
            }
            float cost = leftSplitBin.bbox.getSurfaceArea() / parentSurfaceArea * leftSplitBin.indices.size()
                        + rightSplitBin.bbox.getSurfaceArea() / parentSurfaceArea * rightSplitBin.indices.size();
            // check if is valid, if yes, compare cost
            int left_size = leftSplitBin.indices.size(), right_size = rightSplitBin.indices.size();
            if (left_size > 0 && right_size > 0) {
                hasValidSplit = true;
                if (cost < min_cost) {
                    min_cost = cost;
                    best_axis = axis;
                    best_split = split_start;
                    bestLeftBin = leftSplitBin;
                    bestRightBin = rightSplitBin;
                }
            }
        }
    }

    // 3. if SAH failed, make parent a leaf, even though it might exceed max_leaf_size
    if (!hasValidSplit || best_axis == -1 || best_split == -1) {
        parent.setAsLeaf(startIndex, size);
        return;
    }

    // 4. else, partition faceIndices, create two children, and assign them as the left and right of parent
    auto partition_point = std::partition(faceIndicesBVH.begin() + startIndex, 
                                          faceIndicesBVH.begin() + startIndex + size,
                                          [&](int index) {
                                              return std::find(bestLeftBin.indices.begin(), 
                                                               bestLeftBin.indices.end(), 
                                                               index) != bestLeftBin.indices.end();
                                          });

    bvhNode leftChild = bvhNode();
    leftChild.bbox = bestLeftBin.bbox;
    leftChild.startIndex = startIndex;
    leftChild.size = bestLeftBin.indices.size();
    
    bvhNode rightChild = bvhNode();
    rightChild.bbox = bestRightBin.bbox;
    rightChild.startIndex = startIndex + bestLeftBin.indices.size();
    rightChild.size = bestRightBin.indices.size();

    parent.left = bvhNodes.size();
    parent.right = bvhNodes.size() + 1;
    parent.is_leaf = false;
    bvhNodes.push_back(leftChild);
    bvhNodes.push_back(rightChild);

    buildBVHRecursive(bvhNodes[bvhNodes.size() - 2], leftChild.startIndex, leftChild.size);
    buildBVHRecursive(bvhNodes[bvhNodes.size() - 1], rightChild.startIndex, rightChild.size);
}

void Scene::validateBVH() {
    int largeLeafCount = 0;
    for (size_t i = 0; i < bvhNodes.size(); i++) {
        bvhNode& node = bvhNodes[i];
        assert(node.size > 0);
        if (!node.is_leaf) {
            assert(node.left != -1 && node.right != -1);
            assert(node.size == bvhNodes[node.left].size + bvhNodes[node.right].size);
        }else{
            assert(node.left == -1 && node.right == -1);
            if (node.size > max_leaf_size) {
                largeLeafCount++;
                std::cout << "Large leaf at index " << i << " with size " << node.size << std::endl;
            }
        }
    }
    std::cout << "BVH is valid, " << largeLeafCount << " large leaves found" << std::endl;
}

