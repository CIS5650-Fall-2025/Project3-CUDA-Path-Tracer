#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace postprocess {
    __host__ __device__ inline void gammaCorrection(glm::vec3& color, float gamma) {
        color = glm::pow(color, glm::vec3(1.f / gamma));
    }
}
