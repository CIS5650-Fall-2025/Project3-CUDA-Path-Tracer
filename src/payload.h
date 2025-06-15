#pragma once
#include "sceneStructs.h"

/**
 * PathPayload
 * -----------
 * A wrapper structure that encapsulates all per-ray information needed
 * during shading and scattering.
 *
 * This includes:
 *  - A pointer to the current PathSegment (ray state, throughput, etc.)
 *  - The intersection and material hit by this ray
 *  - The pixel index in the image buffer (for writing outputs)
 *  - The current bounce depth
 *  - Output buffers for denoising (normals and albedo at first bounce)
 * 
 *  * PathPayload is a temporary, per-thread wrapper used only during a single bounce
 * in the shadeMaterial kernel. It exists for the lifetime of the current function call
 * and is passed to scatterRay() to keep the code clean and organized.
 *
 * Note: PathPayload itself is *not* persisted between bounces or kernel launches.
 *
 * What *does* persist across bounces:
 *   - PathSegment* dev_paths:
 *       Holds ray state: origin, direction, throughput (color), remainingBounces
 *   - ShadeableIntersection* dev_intersections:
 *       Stores t-value, surface normal, uv, material ID for the current hit
 *   - glm::vec3* dev_normal and dev_albedo:
 *       Global G-buffer outputs (first-bounce only), used for denoising
 *
 * PathPayload simply wraps all relevant per-ray state (including these global buffers)
 * into one temporary object for easier logic and less argument clutter.
 *
 */

struct PathPayload {
    PathSegment* path;
    ShadeableIntersection intersection;
    Material material;
    int pixelIdx;
    int bounceDepth;

    // First-bounce outputs
    bool firstBounceRecorded = false;
    glm::vec3* gNormalBuffer = nullptr;
    glm::vec3* gAlbedoBuffer = nullptr;

    __host__ __device__ void recordFirstBounce(const glm::vec3& normal, const glm::vec3& albedo) {
        if (bounceDepth == 1 && !firstBounceRecorded) {
            gNormalBuffer[pixelIdx] = normal;
            gAlbedoBuffer[pixelIdx] = albedo;
            firstBounceRecorded = true;
        }
    }
};

