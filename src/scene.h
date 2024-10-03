#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "glm/glm.hpp"
#include "sceneStructs.h"
#include "utilities.h"

using namespace std;

class Scene {
 private:
  ifstream fp_in;
  void loadFromJSON(const std::string& jsonName);
  void Scene::loadMeshesFromGLTF(const std::string& filename, std::vector<Geom>& geoms,
                                 std::vector<Triangle>& triangles, std::vector<Material>& materials);
  void Scene::loadMeshesFromGLTF(const std::string& filename, std::vector<Geom>& geoms,
                                 std::vector<Triangle>& triangles, std::vector<Material>& materials,
                                 const glm::vec3& translate, const glm::vec3& rotate, const glm::vec3& scale);

 public:
  Scene(string filename);
  ~Scene();

  std::vector<Geom> geoms;
  std::vector<Material> materials;
  std::vector<Triangle> triangles;
  RenderState state;
};
