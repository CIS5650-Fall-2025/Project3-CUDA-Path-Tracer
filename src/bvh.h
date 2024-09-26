#pragma once
#include "scene.h"

#define BVH_BNI_CNT 20
#define BVH_LEAF_MAX_CNT 4

struct BVHBin
{
	uint32_t cnt = 0;
	AABB bbox;
};

class BVH
{
public:

	static std::vector<MTBVHNode> buildBVH(Scene& scene, std::vector<AABB>& arr, uint32_t primNum, uint32_t triNum, uint32_t& size);

private:
	static void fillAABBArray(BVHNode* root, std::vector<AABB>& arr);
	static BVHNode* buildBVHRecursiveSAH(std::vector<Primitive>& primitives, uint32_t start, uint32_t end, uint32_t& size);
	static void flattenBVH(std::vector<MTBVHNode>& flattenNodes, BVHNode* root, const uint32_t treeSize);
	static uint32_t flattenBVHResursive(MTBVHNode* flattenNodes, BVHNode* root, uint32_t curr, int dir);
};

