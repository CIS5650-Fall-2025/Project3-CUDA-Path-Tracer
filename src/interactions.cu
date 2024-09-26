#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void coordinateSystem(const glm::vec3 v1, glm::vec3 &v2, glm::vec3 &v3) {
    if (abs(v1.x) > abs(v1.y)) {
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }
    else {
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
            
    v3 = cross(v1, v2);
}

__host__ __device__ glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__host__ __device__ glm::mat3 WorldToLocal(glm::vec3 nor) {
    return transpose(LocalToWorld(nor));
}

// Calculate colours
__host__ __device__ glm::vec3 evalDiffuse(const glm::vec3 albedo){
    return albedo * INV_PI;
}

__host__ __device__ glm::vec3 evalMirror(){
    // Nori's implementation
    return glm::vec3(0.0f);
}

// Calculate PDFs
/** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the angle between the normal and v */
__host__ __device__ float cosTheta(const glm::vec3 &v) {
    return v.z;
}

__host__ __device__ float pdfDiffuse(const glm::vec3 wiL){
    return INV_PI * cosTheta(wiL);
}

__host__ __device__ float pdfMirror(){
    return 0.0f;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 woW,
    glm::vec3 normal, // Here normal is in world space
    glm::vec3 &wiW,
    float &pdf,
    glm::vec3 &c,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 woL = WorldToLocal(normal) * woW; 
    if (m.type == DIFFUSE) {
        wiW = calculateRandomDirectionInHemisphere(normal, rng);

        glm::vec3 wiL = WorldToLocal(normal) * wiW;
        if (cosTheta(woL) <= 0 || cosTheta(wiL) <= 0) {
            pdf = 0.0f;
        }
        else {
            pdf = pdfDiffuse(wiL);
        }

        c = evalDiffuse(m.color);
    }
    else if (m.type == MIRROR) {
        wiW = glm::reflect(woW, normal);
        pdf = pdfMirror();
        c = evalMirror();
    }
}
