#pragma once

struct SelectionState {
    bool enabled = false;
    bool changed = false; // toggle enable mode or change the selected obj

    bool isShellActive = false;

    int pickedID = -1;
    int outlineMaterialID = 0;
    float outlineScale = 0.5f;

    void reset() {
        changed = false;
        pickedID = -1;
        isShellActive = false;
    }
};

extern SelectionState selection;
