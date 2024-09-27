#include "bvh.h"

#include <stack>

#include "utilities.h"

struct BuildTask {
	BuildTask(int s, int r, int n) : start(s), range(r), node(n) {}
	int start;
	int range;
	int node;
};

struct Buckets {
    BBox bbox;
    int numGeoms;
};

struct Partition {
    BBox partitionLeft;
    BBox partitionRight;
    int partitionAxis;
    float partitionPoint;
    float sah;
};

BVH::BVH(std::vector<Geom>&& geoms_, int leafSize) {
    nodes.resize(0);
	build(std::move(geoms_), leafSize);
}

void BVH::clear() {
	nodes.clear();
	geoms.clear();
}

void BVH::build(std::vector<Geom>&& geoms_, int leafSize) {
	nodes.clear();
	geoms = std::move(geoms_);

	if (geoms.empty()) {
		return;
	}

	rootIdx = newNode();
	std::stack<BuildTask> bstack;
	bstack.push(BuildTask(0, geoms.size(), rootIdx));

    while(!bstack.empty()) {
        BuildTask bdata = bstack.top();
		bstack.pop();
        
		Node& node = nodes[bdata.node];

		BBox bbox;
		for (size_t i = 0; i < bdata.range; i++) {
			bbox.enclose(geoms[bdata.start + i].bbox());
		}
		node.bbox = bbox;
		node.start = bdata.start;
		node.size = bdata.range;

        if (!USE_BVH) return;

        // If number of geoms less or equal to leaf size, terminate
        if (node.size <= leafSize) continue;

        // Compute max number of buckets we need based on extent
        glm::vec3 extent = bbox.max - bbox.min;
        int numBuckets = (int)std::max(8.f,
                                        std::max(std::max(floorf(log2f(extent[0])), 
                                                          floorf(log2f(extent[1]))), 
                                                 floorf(log2f(extent[2]))));

        std::vector<Buckets> buckets(numBuckets);

        // Default partition is no partition
        Partition bestPartition;
        bestPartition.partitionAxis = -1;

        for (int axis = 0; axis < 3; axis++) {
            extent = bbox.max - bbox.min;
            // ignore flat axis
			if (extent[axis] < EPSILON) continue;

            // Reset buckets for new iteration
            for (size_t i = 0; i < numBuckets; ++i) {
				buckets[i].bbox = BBox();
				buckets[i].numGeoms = 0;
			}

            // Populate buckets
            float bucketWidth = extent[axis] / (float)numBuckets;
            for (int i = bdata.start; i < bdata.start + bdata.range; i++) {
				Geom& g = geoms[i];
				BBox gBbox = g.bbox();
				int index = (int)((gBbox.center()[axis] - bbox.min[axis]) / bucketWidth);
				index = index < 0 ? 0 : index > numBuckets - 1 ? numBuckets - 1 : index;
				buckets[index].bbox.enclose(gBbox);
				buckets[index].numGeoms++;
			}

            // Consider each splitting scheme using SAH
            for (int idx = 1; idx < numBuckets; idx++) {
                // Left partition
                int N1 = 0;
				BBox B1;
				for (int i = 0; i < idx; ++i) {
					B1.enclose(buckets[i].bbox);
					N1 += buckets[i].numGeoms;
				}
                
                // Right partition
				size_t N2 = 0;
				BBox B2;
				for (int i = idx; i < numBuckets; ++i) {
					B2.enclose(buckets[i].bbox);
					N2 += buckets[i].numGeoms;
				}

				// sah cost & actual split value
				float sah = N1 * B1.surfaceArea() + N2 * B2.surfaceArea();
                if (sah < bestPartition.sah) {
                    bestPartition.partitionLeft = B1;
                    bestPartition.partitionRight = B2;
                    bestPartition.partitionAxis = axis;
                    bestPartition.partitionPoint = bbox.min[axis] + idx * bucketWidth;
                    bestPartition.sah = sah;
                }
            }
        }

        // If there is no paritioning scheme that is better than no partitioning
        // aka all bbox's are concentric, arbitrarily split at halfway point
        int startl = bdata.start;
        int rangel = bdata.range / 2;
        int startr = startl + rangel;
        int ranger = bdata.range - rangel;

        // Otherwise, actually perform partitioning based on bestPatition
        if (bestPartition.partitionAxis > -1) {
            auto it = std::partition(geoms.begin() + bdata.start,
                                    geoms.begin() + bdata.start + bdata.range,
                                    [&bestPartition](Geom& g) {
                                        return g.bbox().center()[bestPartition.partitionAxis] < bestPartition.partitionPoint;
                                    });
                                    
            rangel = std::distance(geoms.begin(), it) - bdata.start;
            startr = startl + rangel;
            ranger = bdata.range - rangel;

            // Shouldn't happen, but just in case
            if (rangel == 0 || ranger == 0) {
                rangel = bdata.range / 2;
                startr = startl + rangel;
                ranger = bdata.range - rangel;
            }
        }

        int lIndex = newNode();
        int rIndex = newNode();

		nodes[bdata.node].l = lIndex;
		nodes[bdata.node].r = rIndex;

		// create new build data
		bstack.push(BuildTask(startl, rangel, nodes[bdata.node].l));
		bstack.push(BuildTask(startr, ranger, nodes[bdata.node].r));
    }
}

int BVH::newNode(BBox bbox, int start, int size, int l, int r) {
	Node n = Node(bbox, start, size, l, r);
	nodes.push_back(n);
    nodeCount++;
	return nodeCount - 1;
}