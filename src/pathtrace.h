#pragma once

#include <vector>
#include "scene.h"

struct RayHasIntersected {
    __host__ __device__ bool operator()(const PathSegment& path) const {
        return path.remainingBounces != 0;
    }
};

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
