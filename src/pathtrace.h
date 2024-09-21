#pragma once

#include <vector>
#include "scene.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_MATERIAL_ID 1

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

struct isRayAlive;
