#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "glm/glm.hpp"
#include "utilities.h"
#include "scene.h"

using namespace std;

struct bbox
{
    glm::vec3 min;
    glm::vec3 max;

    bbox() : min(glm::vec3(std::numeric_limits<float>::max())), max(glm::vec3(std::numeric_limits<float>::lowest())) {}
    bbox(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {
        checkZeroVolume();
    }
    bbox(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
        enclosePoint(a);
        enclosePoint(b);
        enclosePoint(c);
        checkZeroVolume();
    }

    void checkZeroVolume() {
        // Add a small epsilon to avoid zero volume for dimensions with zero extent
        const float epsilon = 1e-7f;
        glm::vec3 extent = this->max - this->min;
        for (int i = 0; i < 3; ++i) {
            if (extent[i] == 0.0f) {
                this->min[i] -= epsilon;
                this->max[i] += epsilon;
            }
        }
    }

    void enclosePoint(const glm::vec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    void encloseBbox(const bbox& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    glm::vec3 getCenter() const {
        return (min + max) * 0.5f;
    }

    float getSurfaceArea() const {
        glm::vec3 extent = max - min;
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }
};

struct bvhNode {
    bbox bbox;
    size_t left; // in scene.bvhNodes
    size_t right;

    size_t startIndex; // in scene.triangleBboxes
    size_t size;

    bool is_leaf;

    bvhNode() : left(-1), right(-1), startIndex(-1), size(-1), is_leaf(false) {}
    size_t endIndex() {
        return startIndex + size;
    }
    void setAsLeaf(size_t startIndex, size_t size){
        this->startIndex = startIndex;
        this->size = size;
        this->is_leaf = true;
        this->left = -1;
        this->right = -1;
    }
};
