#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

class GuiDataContainer
{
public:
    GuiDataContainer()
        : TracedDepth(0), EnableRayCompaction(true),
        EnableMaterialSort(false), EnableDenoise(true), EnableAntialiasing(true),
        EnableDOF(false), InitialLensRadius(0.05f), InitialFocalDist(3.5f),
        LensRadius(0.05f), FocalDist(3.5f) {}

    int TracedDepth;

    // GUI Toggles
    bool EnableRayCompaction;
    bool EnableMaterialSort;
    bool EnableDenoise;
    bool EnableAntialiasing;
    bool EnableDOF;

    // Depth of Field
    float InitialLensRadius;  // from scene.json
    float InitialFocalDist;
    float LensRadius;         // user-controlled in GUI
    float FocalDist;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
