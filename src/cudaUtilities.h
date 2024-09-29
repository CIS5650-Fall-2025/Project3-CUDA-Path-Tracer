#pragma once
#include "utilities.h"
#include "sceneStructs.h"

extern Triangle* dev_triangles;
extern Geom* dev_geoms;
extern int dev_numGeoms;
extern Material* dev_materials;
extern int* dev_triTransforms;

void updateSceneTrianglesTransform(Geom* geoms, Triangle* triangles, int numGeoms, int numTriangles);

void initSceneCuda(Geom* geoms, Material* materials, Triangle* triangles, int numGeoms, int numMaterials, int numTriangles);
__global__ void updateTriangleTransformIndex(Geom* dev_geoms, int* dev_triTransforms, int numGeoms);
__global__ void updateTriangleTransform(Geom* dev_geoms, Triangle* dev_triangles, int* dev_triTransforms, int numGeoms, int numTriangles);

//void updateTrianglesTransform(Geom* geom, Triangle* triangles, int numGeoms, int numTriangles);
void freeSceneCuda();

void printGeoms();

void checkCUDAError(const char* msg);