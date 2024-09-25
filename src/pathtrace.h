#pragma once

#include <vector>
#include "scene.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_MATERIAL_ID 1
#define BVH 1
#define OIDN 1

#define PI_OVER_TWO       1.5707963267948966192313216916397514420986f
#define PI_OVER_FOUR      0.7853981633974483096156608458198757210493f

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u);
__device__ glm::vec3 checkerboard(glm::vec2 uv);
__device__ glm::vec3 palettes(glm::vec2 uv);

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
#if BVH
    BVHNode* bvh ,
#endif
    Mesh* meshes
    , glm::vec3* vertices
    , glm::vec3* normals
    , glm::vec2* texcoords);

struct isRayAlive;
