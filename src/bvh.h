#pragma once

#include "sceneStructs.h"

#include "bbox.h"

const bool USE_BVH = true;

// BVH implementation was heavily by Scotty3D
// https://github.com/CMU-Graphics/Scotty3D
class BVH {
public:
    class Node {
    public:
        Node(BBox bbox_ = {}, int start_ = 0 , int size_ = 0, int l_ = 0, int r_ = 0) :
            bbox(bbox_), start(start_), size(size_), l(l_), r(r_) {}

        BBox bbox;
        // A node is a leaf if left and right children are same
        int start, size, l, r;
    };

    BVH() = default;
    BVH(std::vector<Geom>&& geoms_, int leafSize);

	void clear();
	void build(std::vector<Geom>&& geoms_, int leafSize = 1);

	std::vector<Geom> geoms;
	std::vector<Node> nodes;
	int rootIdx = 0;

private:
	int newNode(BBox box = {}, int start = 0, int size = 0, int l = 0, int r = 0);
};