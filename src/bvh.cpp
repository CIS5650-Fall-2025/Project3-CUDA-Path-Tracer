#pragma once

#include "bvh.h"
#include "sceneStructs.h"

using namespace std;

struct bbox;
struct bvhNode;

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
    }
}

