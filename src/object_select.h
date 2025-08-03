#pragma once

#include "scene.h"
#include "intersections.h"
#include "glm/glm.hpp"


struct SelectionState {
    bool enabled = false;
    bool changed = false; // toggle enable mode or change the selected obj

    int pickedID = -1;
    int previousPickedID = -1;

    int outlineMaterialID = 0;
    float outlineThickness = 0.5f;

    void reset() {
        changed = false;
        pickedID = -1;
    }
};

extern SelectionState selection;


// Host interface
int pickObject(int mouseX, int mouseY, Scene* scene, Geom* dev_geoms);

void addHighlightShell(int pickedID, Scene* scene);
void removeHighlightShell(Scene* scene);