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
}

struct Partition {
	Partition(BBox pL, BBox pR, int pA, float pP, float sah_) : 
        partitionLeft(pL), partitionRight(pR), partitionAxis(pA), partitionPoint(pP), sah(sah_) {}
    BBox partitionLeft;
    BBox partitionRight;
    int partitionAxis;
    float partitionPoint;
    float sah;
}

BVH::BVH(std::vector<Geom>&& geoms_, int leafSize) {
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

	rootIdx = 0;
	std::stack<BuildTask> bstack;
	bstack.push(BuildTask(0, geoms.size(), newNode()));

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

        // If number of geoms less or equal to leaf size, terminate
        if (bdata.range <= leafSize) continue;

        // Compute max number of buckets we need based on extent
        glm::vec3 extent = bbox.max - bbox.min;
        float numBuckets = std::max(std::max(powf(floorf(log2f(extent[0]))), 
                                            powf(floorf(log2f(extent[1])))), 
                                    powf(floorf(log2f(extent[2]))));

        std::vector<Buckets> buckets(numBuckets);

        // Default partition is no partition
        Partition bestPartition = Partition(BBox(), BBox(), -1, 0.0f, bbox.surfaceArea() * bdata.range);

        for (int axis = 0; axis < 3; axis++) {
            extent = bbox.max - bbox.min;
            // ignore flat axis
			if (extent[axis] < EPSILON) continue;

            // Reset buckets for new iteration
            for (size_t i = 0; i < numBuckets; ++i) {
				buckets[i].bb = BBox();
				buckets[i].num_prims = 0;
			}

            // Populate buckets
            float bucketWidth = extent[axis] / numBuckets;
            for (int i = bdata.start; i < bdata.start + bdata.range; i++) {
				Geom& g = geoms[i];
				BBox gBbox = g.bbox();
				int index = (int)((gBbox.center()[axis] - bbox.min[axis]) / bucketWidth);
				index = std::clamp(index, 0, numBuckets - 1);
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
				for (int i = idx; i < num_buckets; ++i) {
					B2.enclose(buckets[i].bbox);
					N2 += buckets[i].numGeoms;
				}

				// sah cost & actual split value
				float sah = N1 * B1.surfaceArea() + N2 * B2.surfaceArea();
                if (sah < bestPartition.sah) 
                    bestPartition = Partition(b1, b2, axis, bbox.min[axis] + idx * bucketWidth, sah);
            }
        }

        int startl = bdata.start;
        int rangel = bdata.range / 2;
        int startr = startl + rangel;
        int ranger = bdata.range - rangel;

        // If there is no paritioning scheme that is better than no partitioning
        // aka all bbox's are concentric, arbitrarily split at halfway point
        if (bestPartition.partitionAxis > -1) {
            // Actually perform partitioning based on bestPatition
            auto it = std::partition(geoms.begin() + bdata.start,
                                    geoms.begin() + bdata.start + bdata.range,
                                    [bestPartition.partitionAxis, bestPartition.partitionPoint](const Geom& g) {
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

		nodes[bdata.node].l = newNode();
		nodes[bdata.node].r = newNode();

		// create new build data
		bstack.push(BuildTask(startl, rangel, nodes[bdata.node].l));
		bstack.push(BuildTask(startr, ranger, nodes[bdata.node].r));
    }
}

int BVH::newNode(BBox box, int start, int size, int l, int r) {
	Node n = Node(bbox, start, size, l, r);
	nodes.push_back(n);
	return nodes.size() - 1;
}