#pragma once

#include "scene.h"
#include "intersections.h"
#include "selection_state.h"
#include "glm/glm.hpp"


// Host interface
int selectObjectFromScreen(
    const Scene& scene,
    const glm::vec2& screenPixel,
    const glm::ivec2& resolution
);


void addHighlightShell(int pickedID, Scene* scene);
void removeHighlightShell(Scene* scene);