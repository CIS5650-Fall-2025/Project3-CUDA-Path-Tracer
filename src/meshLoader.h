#pragma once

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "utilities.h"
#include "sceneStructs.h"

bool endsWith(const std::string& str, const std::string& suffix);
void loadOBJ(const std::string &filepath, std::vector<Triangle> &faces, std::vector<glm::vec3> &verts, std::vector<glm::vec3> &normals, std::vector<int> &indices);
void loadGLTFOrGLB(const std::string &filepath, std::vector<Triangle> &faces);