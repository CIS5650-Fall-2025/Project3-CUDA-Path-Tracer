#pragma once

#include "utilities.h"
#include <cuda_runtime.h>

namespace math
{

    __device__ static glm::vec2 sampleUniformDisk(float x, float y) {
        float theta = y * TWO_PI;
        return glm::vec2(glm::cos(theta), glm::sin(theta)) * glm::sqrt(x);
    }

    __device__ static glm::vec3 sampleHemisphereCosine(float x, float y) {
        glm::vec2 d = sampleUniformDisk(x, y);
        return glm::vec3(d, glm::sqrt(1.f - glm::dot(d, d)));
    }

    __device__ inline float clampCos(const glm::vec3& w) { return glm::max(w.z, 0.f); }

    __device__ inline float absDot(const glm::vec3& a, const glm::vec3& b) { return glm::abs(glm::dot(a, b)); }

    __device__ inline float cosTheta(const glm::vec3& w) { return w.z; }

    __device__ inline float cos2Theta(const glm::vec3& w) { return w.z * w.z; }

    __device__ inline float absCosTheta(const glm::vec3& w) { return glm::abs(w.z); }

    __device__ inline float sin2Theta(const glm::vec3& w) { return glm::max(0.f, 1.f - cos2Theta(w)); }

    __device__ inline float sinTheta(const glm::vec3& w) { return glm::sqrt(sin2Theta(w)); }

    __device__ inline float tanTheta(const glm::vec3& w) { return sinTheta(w) / cosTheta(w); }

    __device__ inline float tan2Theta(const glm::vec3& w) { return sin2Theta(w) / cos2Theta(w); }

    __device__ inline float cosPhi(const glm::vec3 w)
    {
        float st = sinTheta(w);
        return (st == 0) ? 1 : glm::clamp(w.x / st, -1.f, 1.f);
    }

    __device__ inline float sinPhi(const glm::vec3 w)
    {
        float st = sinTheta(w);
        return (st == 0.f) ? 0.f : glm::clamp(w.y / st, -1.f, 1.f);
    }
    __device__ inline float cos2Phi(const glm::vec3 w) { return cosPhi(w) * cosPhi(w); }

    __device__ inline float sin2Phi(const glm::vec3 w) { return sinPhi(w) * sinPhi(w); }




    // ref: http://marc-b-reynolds.github.io/quaternions/2016/07/06/Orthonormal.html
    __device__ static glm::mat3 getTBN(const glm::vec3& N)
    {
        float x = N.x, y = N.y, z = N.z;
        float sz = z < 0.f ? -1.f : 1.f;
        float a = y / (z + sz);
        float b = y * a;
        float c = x * a;
        glm::vec3 T = glm::vec3(-z - b, c, x);
        glm::vec3 B = glm::vec3(sz * c, sz * b - 1, sz * y);

        return glm::mat3(T, B, N);
    }
}

