#pragma once
#include "utilities.h"
#include "sceneStructs.h"

//static Triangle* dev_triangles = NULL;
//static Geom* dev_geoms = NULL;
//static Material* dev_materials = NULL;

//__global__ void initSceneCuda(Geom* geoms, Material* materials, int numGeoms, int numMaterials);
__global__ void updateTriangleTransform(Geom& geom, Triangle* triangles, int numTriangles);

void updateTrianglesTransform(Geom& geom, Triangle* triangles);
