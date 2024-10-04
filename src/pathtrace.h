#pragma once

#include <vector>
#include "scene.h"
#include <OpenImageDenoise/oidn.hpp>

#define STREAMCOMPACTION 0
#define SORTBYMATERIAL 0

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, 
		oidn::FilterRef& oidn_filter,
		int frame, int iteration);
