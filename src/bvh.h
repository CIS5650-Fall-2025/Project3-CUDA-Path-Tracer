#pragma once
#include "utilities.h"
#include "sceneStructs.h"
#include "memoryArena.h"


class BVHAccel
{
private:
	std::vector<std::shared_ptr<Triangle>> primitives;
	const int maxPrimsInNode;

	struct AABB
	{
		glm::vec3 min;
		glm::vec3 max;
		static AABB Union(const AABB& b1, const AABB& b2)
		{
			AABB ret;
			ret.min = glm::min(b1.min, b2.min);
			ret.max = glm::max(b1.max, b2.max);
			return ret;
		}

		static AABB Union(const AABB& b, const glm::vec3& p)
		{
			AABB ret;
			ret.min = glm::min(b.min, p);
			ret.max = glm::max(b.max, p);
			return ret;
		}

		int maxExtent() const
		{
			glm::vec3 diag = max - min;
			if (diag.x > diag.y && diag.x > diag.z)
				return 0;
			else if (diag.y > diag.z)
				return 1;
			else
				return 2;
		}

		glm::vec3 Offset(const glm::vec3& p) const
		{
			glm::vec3 o = p - min;
			if (max.x > min.x) o.x /= max.x - min.x;
			if (max.y > min.y) o.y /= max.y - min.y;
			if (max.z > min.z) o.z /= max.z - min.z;
			return o;
		}

		float SurfaceArea() const
		{
			glm::vec3 d = max - min;
			return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
		}
	};

	struct BVHPrimitiveInfo
	{
		int primitiveNumber;
		glm::vec3 centroid;
		AABB bounds;
	};

	struct BVHBuildNode
	{
		AABB bounds;
		BVHBuildNode* children[2];
		int splitAxis;
		int firstPrimOffset;
		int nPrimitives;
		void initLeaf(int first, int n, const AABB& b)
		{
			firstPrimOffset = first;
			nPrimitives = n;
			bounds = b;
			children[0] = children[1] = nullptr;
		}
		void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1)
		{
			children[0] = c0;
			children[1] = c1;
			bounds = AABB::Union(c0->bounds, c1->bounds);
			splitAxis = axis;
			nPrimitives = 0;
		}

	};

	BVHBuildNode* recursiveBuild(MemoryArena& arena,
		std::vector<BVHPrimitiveInfo>& primitiveInfo, int start,
		int end, int* totalNodes,
		std::vector<std::shared_ptr<Triangle>>& orderedPrims) {
		BVHBuildNode* node = arena.Alloc<BVHBuildNode>();
		(*totalNodes)++;

		// compute bounds of all primitives in BVH node
		AABB bounds;
		for (int i = start; i < end; ++i)
			bounds = AABB::Union(bounds, primitiveInfo[i].bounds);

		int nPrimitives = end - start;
		if (nPrimitives == 1) {
			// create leaf node
			int firstPrimOffset = orderedPrims.size();
			for (int i = start; i < end; ++i) {
				int primNum = primitiveInfo[i].primitiveNumber;
				orderedPrims.push_back(primitives[primNum]);
			}
			node->initLeaf(firstPrimOffset, nPrimitives, bounds);
			return node;
		}
		else {
			// Compute bound of primitive centroids, choose split dimension dim
			AABB centroidBounds; // initialize?
			for (int i = start; i < end; ++i)
				centroidBounds = AABB::Union(centroidBounds, primitiveInfo[i].centroid);
			int dim = centroidBounds.maxExtent();

			//Partition primitives into two sets and build children
			int mid = (start + end) / 2;
			if (centroidBounds.max[dim] == centroidBounds.min[dim]) {
				// Create leaf BVHBuildNode 
				int firstPrimOffset = orderedPrims.size();
				for (int i = start; i < end; ++i) {
					int primNum = primitiveInfo[i].primitiveNumber;
					orderedPrims.push_back(primitives[primNum]);
				}
				node->initLeaf(firstPrimOffset, nPrimitives, bounds);
				return node;
			}
			else {
				//// equal count partition
				//mid = (start + end) / 2;
				//std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
				//	&primitiveInfo[end - 1] + 1,
				//	[dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
				//		return a.centroid[dim] < b.centroid[dim];
				//	});

				// SAH partition
				if (nPrimitives <= 4) {
					// Partition primitives into equally sized subsets
					mid = (start + end) / 2;
					std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
						&primitiveInfo[end - 1] + 1,
						[dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
							return a.centroid[dim] < b.centroid[dim];
						});
				}
				else {
					// Allocate BucketInfo for SAH partition buckets
					constexpr int nBuckets = 12;
					struct BucketInfo {
						int count = 0;
						AABB bounds;
					};
					BucketInfo buckets[nBuckets];

					// Initialize BucketInfo for SAH partition buckets
					for (int i = start; i < end; ++i) {
						int b = nBuckets *
							centroidBounds.Offset(primitiveInfo[i].centroid)[dim];
						if (b == nBuckets) b = nBuckets - 1;
						buckets[b].count++;
						buckets[b].bounds = AABB::Union(buckets[b].bounds, primitiveInfo[i].bounds);
					}

					// Compute costs for splitting after each bucket
					float cost[nBuckets - 1];
					for (int i = 0; i < nBuckets - 1; ++i) {
						AABB b0, b1;
						int count0 = 0, count1 = 0;
						for (int j = 0; j <= i; ++j) {
							b0 = AABB::Union(b0, buckets[j].bounds);
							count0 += buckets[j].count;
						}
						for (int j = i + 1; j < nBuckets; ++j) {
							b1 = AABB::Union(b1, buckets[j].bounds);
							count1 += buckets[j].count;
						}
						cost[i] = 0.125f + (count0 * b0.SurfaceArea() +
							count1 * b1.SurfaceArea()) / bounds.SurfaceArea();
					}

					// Find bucket to split at that minimizes SAH metric
					float minCost = cost[0];
					int minCostSplitBucket = 0;
					for (int i = 1; i < nBuckets - 1; ++i) {
						if (cost[i] < minCost) {
							minCost = cost[i];
							minCostSplitBucket = i;
						}
					}

					// Either create leaf or interior BVHBuildNode
					float leafCost = nPrimitives;
					if (nPrimitives > maxPrimsInNode || minCost < leafCost) {
						BVHPrimitiveInfo* pmid = std::partition(&primitiveInfo[start],
							&primitiveInfo[end - 1] + 1,
							[=](const BVHPrimitiveInfo& pi) {
								int b = nBuckets * centroidBounds.Offset(pi.centroid)[dim];
								if (b == nBuckets) b = nBuckets - 1;
								return b <= minCostSplitBucket;
							});
						mid = pmid - &primitiveInfo[0];
					}
					else {
						// Create leaf BVHBuildNode
						int firstPrimOffset = orderedPrims.size();
						for (int i = start; i < end; ++i) {
							int primNum = primitiveInfo[i].primitiveNumber;
							orderedPrims.push_back(primitives[primNum]);
						}
						node->initLeaf(firstPrimOffset, nPrimitives, bounds);
						return node;
					}

					// Partition primitives based on splitMethod
					node->InitInterior(dim,
						recursiveBuild(arena, primitiveInfo, start, mid,
							totalNodes, orderedPrims),
						recursiveBuild(arena, primitiveInfo, mid, end,
							totalNodes, orderedPrims));
				}
				return node;
			}
		}
	}

};