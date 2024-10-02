#pragma once

#include <vector>
#include "scene.h"

#define STREAMCOMPACTION 0
#define SORTBYMATERIAL 0

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
