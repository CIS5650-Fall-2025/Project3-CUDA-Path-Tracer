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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &mat,
    thrust::default_random_engine &rng)
    {
    if (mat.hasReflective > 0.0f || mat.hasRefractive > 0.0f) {
        if (mat.hasReflective > 0.0f && mat.hasRefractive == 0.0f) {
            // perfect specular reflection
            glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
            
            // Update the ray origin and direction
            pathSegment.ray.origin = intersect;
            pathSegment.ray.direction = glm::normalize(reflectedDir);
            pathSegment.color *= mat.color;
        } else if (mat.hasReflective == 0.0f && mat.hasRefractive > 0.0f) {
            // transmissive
            float reflectionProbability = thrust::uniform_real_distribution<float>(0, 1)(rng);
            float cosThetaI = glm::dot(-pathSegment.ray.direction, normal);
            float ior = mat.indexOfRefraction;
            glm::vec3 fresnel = fresnelSchlick(cosThetaI, ior);
            bool entering = glm::dot(pathSegment.ray.direction, normal) < 0;
            normal = entering ? normal : -normal;
            float eta = entering ? (1.0f / ior) : ior ; // Adjust based on entering or leaving glass
            if (fresnel.x < mat.hasRefractive){
                // refraction
                glm::vec3 refractedDir = glm::refract(pathSegment.ray.direction, normal, eta);
                if (glm::length(refractedDir) == 0.0f){
                    // Total internal reflection
                    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
                    pathSegment.color *= mat.color;
                }else{
                    pathSegment.ray.direction = glm::normalize(refractedDir);
                    pathSegment.color *= mat.color;
                }
            }else{
                // reflection
                glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
                pathSegment.ray.direction = glm::normalize(reflectedDir);
                pathSegment.color *= mat.color;
            }
            pathSegment.ray.origin = intersect - 0.001f * normal; // why 0.0001 doesn't work?

        } else if (mat.hasReflective > 0.0f && mat.hasRefractive > 0.0f) {
            // glass like
            float reflectionProbability = thrust::uniform_real_distribution<float>(0, 1)(rng);
            if ( reflectionProbability < mat.hasReflective) {
                // Specular reflection
                glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
                pathSegment.ray.direction = glm::normalize(reflectedDir);
                pathSegment.color *= mat.color;
            } else {
                // Refraction
                float cosThetaI = glm::dot(-pathSegment.ray.direction, normal);
                float ior = mat.indexOfRefraction;
                glm::vec3 fresnel = fresnelSchlick(cosThetaI, ior);
                bool entering = glm::dot(pathSegment.ray.direction, normal) < 0;
                normal = entering ? normal : -normal;
                float eta = entering ? (1.0f / ior) : ior ; // Adjust based on entering or leaving glass
                if (fresnel.x < mat.hasRefractive){
                    // refraction
                    glm::vec3 refractedDir = glm::refract(pathSegment.ray.direction, normal, eta);
                    if (glm::length(refractedDir) == 0.0f){
                        // Total internal reflection
                        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
                        pathSegment.color *= mat.color;
                    }else{
                        pathSegment.ray.direction = glm::normalize(refractedDir);
                        pathSegment.color *= mat.color;
                    }
                }else{
                    // reflection
                    glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
                    pathSegment.ray.direction = glm::normalize(reflectedDir);
                    pathSegment.color *= mat.color;
                }
                
            }
            pathSegment.ray.origin = intersect - 0.001f * normal; // why 0.0001 doesn't work?
        }
    }else{
        // diffuse
        glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = newDir;
        float cosTheta = glm::dot(pathSegment.ray.direction, normal);
        float lambert = max(0.0f, cosTheta);
        float pdf = 0.0f;
        if (lambert <= 1.0f) {
            pdf = lambert / PI;
        }
        glm::vec3 diffuseColor = mat.color / PI;
        pathSegment.color *= diffuseColor * lambert / pdf;
    }
}

__host__ __device__ glm::vec3 fresnelSchlick(float cosThetaI, float etaT) {

    float etaI = 1.0f; // Index of refraction for air
    float R = (1.0f - etaI) / (1.0f + etaI);
    float R0 = R * R;

    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    float ret = R0 + (1.0f - R0) * pow(1.0f - cosThetaI, 5.0f);

    return glm::vec3(ret);
}
