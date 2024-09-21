#pragma once

#include <vector>
#include "scene.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_MATERIAL_ID 1

#define PI_OVER_TWO       1.5707963267948966192313216916397514420986f
#define PI_OVER_FOUR      0.7853981633974483096156608458198757210493f

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

struct isRayAlive;
