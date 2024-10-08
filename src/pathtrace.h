#pragma once

#include <vector>
#include "scene.h"
#include <OpenImageDenoise/oidn.hpp>

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4* pbo,
		oidn::FilterRef& oidn_filter,
		float& percentD,
		int frame, int iteration);