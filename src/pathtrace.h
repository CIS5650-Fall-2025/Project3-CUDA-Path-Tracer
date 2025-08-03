#pragma once

#include <vector>
#include "scene.h"
#include "sceneStructs.h"

#include "texture_utils.h"
#include "texture.h"

#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "texture_utils.h"
#include "scene.h"
#include "denoise.h"
#include "object_select.h"


Geom* getDevGeoms();

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
