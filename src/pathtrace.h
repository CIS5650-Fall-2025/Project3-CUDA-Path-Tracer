#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void copyBVHNodes(const std::vector<BVHNode>& bvh, BVHNode* dev_bvhNodes);
void freeBVHNode(BVHNode* dev_bvhNodes);