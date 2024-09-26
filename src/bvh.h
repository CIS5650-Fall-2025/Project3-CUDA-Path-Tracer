#pragma once

#include "sceneStructs.h"

#include "bbox.h"

class BVH {
public:
    class Node {
    public:
        Node(BBox bbox_, int start_, int size_, int l_, int r_) : 
            bbox(bbox_), start(start_), size(size_), l(l_), r(r_) {}
        BBox bbox;
        int start, size, l, r;

        // A node is a leaf if left and right children are same
        bool isLeaf() const { return l == r; }
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