#include "bvh.h"

const static int dirs[] = { 1, -1, 2, -2, 3, -3 };


std::vector<MTBVHNode> BVH::buildBVH(Scene& scene, std::vector<AABB>& arr, uint32_t primNum, uint32_t triNum, uint32_t& size)
{
	size = 0;
	BVHNode* root = buildBVHRecursiveSAH(scene.primitives, 0, scene.primitives.size(), size);
	arr.resize(size);
	fillAABBArray(root, arr);
	std::vector<MTBVHNode> flattenNodes;
	flattenBVH(flattenNodes, root, size);
	return flattenNodes;
}

void BVH::fillAABBArray(BVHNode* root, std::vector<AABB>& arr)
{
	if (!root) return;
	arr[root->nodeID] = root->bbox;
	fillAABBArray(root->left, arr);
	fillAABBArray(root->right, arr);
}


BVHNode* BVH::buildBVHRecursiveSAH(std::vector<Primitive>& primitives, uint32_t start, uint32_t end, uint32_t& size)
{
	BVHNode* root = new BVHNode();
	root->nodeID = size;
	++size;

	AABB bBox;
	AABB cBox;

	for (uint32_t i = start; i < end; ++i)
	{
		cBox.expand(primitives[i].bbox.center());
		bBox.expand(primitives[i].bbox);
	}

	root->bbox = bBox;
	root->startPrimID = UINT32_MAX;
	root->endPrimID = UINT32_MAX;
	int axis = bBox.majorAxis();
	glm::vec3 cExtend = cBox.extend();

	// sort alone major axis
	std::sort(primitives.begin() + start, primitives.begin() + end, [axis](const Primitive& p1, const Primitive& p2) {
		return p1.bbox.center()[axis] < p2.bbox.center()[axis];
		});

	// leaf node
	if (end - start < BVH_LEAF_MAX_CNT || cExtend[axis] < 0.001f)
	{
		root->left = nullptr;
		root->right = nullptr;
		root->startPrimID = start;
		root->endPrimID = end;
		return root;
	}

	float totalSurfaceArea = bBox.surfaceArea();
	float cMin = cBox.bmin[axis];

	BVHBin bins[BVH_BNI_CNT];
	for (int i = start; i < end; i++)
	{
		int bin = (primitives[i].bbox.center()[axis] - cMin) / cExtend[axis] * BVH_BNI_CNT;
		if (bin == BVH_BNI_CNT) --bin;

		bins[bin].cnt += 1;
		bins[bin].bbox.expand(primitives[i].bbox);
	}

	float minCost = FLT_MAX;
	int minDiv = 0;
	int leftCnt = 0;
	int rightCnt = 0;
	for (int i = 0; i <= BVH_BNI_CNT; ++i)
	{
		AABB leftBox;
		AABB rightBox;
		leftCnt = 0;
		rightCnt = 0;
		for (int j = 0; j < i; ++j)
		{
			leftCnt += bins[j].cnt;
			leftBox.expand(bins[j].bbox);
		}
		for (int j = i; j < BVH_BNI_CNT; ++j)
		{
			rightCnt += bins[j].cnt;
			rightBox.expand(bins[j].bbox);
		}
		float cost = ((float)leftCnt * leftBox.surfaceArea() + (float)rightCnt * rightBox.surfaceArea()) / totalSurfaceArea;
		if (cost < minCost)
		{
			minCost = cost;
			minDiv = i;
		}
	}

	if (minDiv == 0 || minDiv == BVH_BNI_CNT)
	{
		root->left = nullptr;
		root->right = nullptr;
		root->startPrimID = start;
		root->endPrimID = end;
	}
	else
	{
		uint32_t mid = start;
		for (int i = 0; i < minDiv; ++i) mid += bins[i].cnt;
		root->left = buildBVHRecursiveSAH(primitives, start, mid, size);
		root->right = buildBVHRecursiveSAH(primitives, mid, end, size);
	}

	return root;
}

void BVH::flattenBVH(std::vector<MTBVHNode>& flattenNodes, BVHNode* root, const uint32_t treeSize)
{
	flattenNodes.resize(6 * treeSize);
	for (int i = 0; i < 6; ++i)
	{
		flattenBVHResursive(&flattenNodes[i * treeSize], root, 0, i);
	}
}

uint32_t BVH::flattenBVHResursive(MTBVHNode* flattenNodes, BVHNode* root, uint32_t curr, int dir)
{
	if (!root) return UINT32_MAX;

	flattenNodes[curr].bboxID = root->nodeID;
	flattenNodes[curr].startPrimID = root->startPrimID;
	flattenNodes[curr].endPrimID = root->endPrimID;

	int axis = glm::abs(dirs[dir]) - 1;
	float sign = (dir % 2) == 0 ? 1.f : -1.f;
	int missNext;

	if (root->left)
	{
		BVHNode* nextNode = root->left;
		BVHNode* missNode = root->right;
		if ((nextNode->bbox.center()[axis] - missNode->bbox.center()[axis]) * sign > 0.f)
		{
			std::swap(nextNode, missNode);
		}

		missNext = flattenBVHResursive(flattenNodes, nextNode, curr + 1, dir);
		missNext = flattenBVHResursive(flattenNodes, missNode, missNext, dir);
		flattenNodes[curr].missNext = missNext;
	}
	else
	{
		missNext = curr + 1;
		flattenNodes[curr].missNext = missNext;
	}
	return missNext;
}
