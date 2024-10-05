#pragma once

#include <glm/glm.hpp>

// BBox implementation was heavily by Scotty3D
// https://github.com/CMU-Graphics/Scotty3D
struct BBox {
    BBox() : min(FLT_MAX), max(-FLT_MAX) {}
	BBox(glm::vec3 min_, glm::vec3 max_) : min(min_), max(max_) {}

    void enclose(glm::vec3 point) {
		min = glm::vec3(std::min(min[0], point[0]), std::min(min[1], point[1]), std::min(min[2], point[2]));
		max = glm::vec3(std::max(max[0], point[0]), std::max(max[1], point[1]), std::max(max[2], point[2]));
	}
	void enclose(BBox box) {
		min = glm::vec3(std::min(min[0], box.min[0]), std::min(min[1], box.min[1]), std::min(min[2], box.min[2]));
		max = glm::vec3(std::max(max[0], box.max[0]), std::max(max[1], box.max[1]), std::max(max[2], box.max[2]));
	}

    glm::vec3 center() const {
		return (min + max) * 0.5f;
	}

    float surfaceArea() const {
        // If invalid or empty bbox, surface area = 0
		if (min.x > max.x || min.y > max.y || min.z > max.z) return 0.0f;

		glm::vec3 extent = max - min;
		return 2.0f * (extent.x * extent.z + extent.x * extent.y + extent.y * extent.z);
	}
    
	BBox& transform(const glm::mat4& trans) {
		glm::vec3 amin = min, amax = max;
		min = max = glm::vec3(trans[3][0], trans[3][1], trans[3][2]);
		for (uint32_t i = 0; i < 3; i++) {
			for (uint32_t j = 0; j < 3; j++) {
				float a = trans[j][i] * amin[j];
				float b = trans[j][i] * amax[j];
				if (a < b) {
					min[i] += a;
					max[i] += b;
				} else {
					min[i] += b;
					max[i] += a;
				}
			}
		}
		return *this;
	}

    glm::vec3 min, max;
};