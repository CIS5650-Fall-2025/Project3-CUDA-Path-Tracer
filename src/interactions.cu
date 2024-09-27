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
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.


    //glm::vec3 ray_dir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));

    //pathSegment.ray.origin = intersect + EPSILON * glm::normalize(ray_dir);

    //pathSegment.ray.direction = ray_dir;

    //pathSegment.color *= m.color;


    thrust::uniform_real_distribution<float> u01(0, 1);

    float lightTerm = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f));
    pathSegment.color *= (m.color * lightTerm) * 0.3f + ((1.0f - intersect.t * 0.02f) * m.color) * 0.7f;
    pathSegment.color *= u01(rng); // apply some noise because why not


    //if (m.hasRefractive) 
    //{
    //    // Calculate refraction
    //    float ior = m.indexOfRefraction;  // Index of refraction, n = c/v;c is the speed of light in a vacuum.
    //    glm::vec3 incident = pathSegment.ray.direction;
    //    glm::vec3 refracted;
    //    glm::vec3 reflection = glm::reflect(incident, normal);

    //    // Determine if we're entering or exiting the material
    //    float cosThetaI = glm::dot(incident, normal);
    //    float eta = (cosThetaI < 0) ? (1.0f / ior) : ior;  // Calculate the ratio of indices
    //    glm::vec3 refractNormal = (cosThetaI < 0) ? normal : -normal;  // Adjust normal


    //    // Recalculate cosThetaI for the possibly flipped normal
    //    cosThetaI = glm::dot(incident, refractNormal); 
    //    float sinThetaTSquared = eta * eta * (1.0f - cosThetaI * cosThetaI);

    //    // Check for total internal reflection
    //    if (sinThetaTSquared > 1.0f)
    //    {
    //        // Total internal reflection: only reflect
    //        pathSegment.ray.origin = intersect + EPSILON * normal;  // Offset to avoid self-intersection
    //        pathSegment.ray.direction = reflection;
    //        pathSegment.color *= m.specular.color;
    //    }
    //    else
    //    {
    //        // Calculate refracted direction
    //        refracted = glm::refract(incident, refractNormal, eta);

    //        // Fresnel effect (Schlick's approximation)
    //        // The Fresnel effect describes how the amount of reflected light changes depending on the angle of incidence.
    //        float R0 = pow((1.0f - ior) / (1.0f + ior), 2.0f);
    //        float fresnelReflectance = R0 + (1.0f - R0) * pow(1.0f - fabs(cosThetaI), 5.0f);


    //        thrust::uniform_real_distribution<float> u01(0, 1);
    //        float reflect_prob = 2.f; //any value larger than 1 

    //        if (m.hasReflective) {
    //            float reflect_prob = u01(rng);
    //        }

    //        if (reflect_prob < fresnelReflectance)
    //        {
    //            // Reflect
    //            pathSegment.ray.origin = intersect + EPSILON * normal;
    //            pathSegment.ray.direction = glm::normalize(reflection);
    //            pathSegment.color *= m.specular.color;  // Reflective color
    //        }
    //        else
    //        {
    //            // Refract
    //            pathSegment.ray.origin = intersect - EPSILON * refractNormal;  // Offset slightly inside the material
    //            pathSegment.ray.direction = glm::normalize(refracted);
    //            pathSegment.color *= m.color;  // Refractive color
    //        }

    //    }
    //}
    //else if (m.hasReflective) //pure reflection
    //{

    //    glm::vec3 incident = pathSegment.ray.direction;
    //    glm::vec3 reflection = glm::reflect(incident, normal);

    //    pathSegment.ray.origin = intersect + EPSILON * normal;  
    //    pathSegment.ray.direction = glm::normalize(reflection);

    //    pathSegment.color *= m.specular.color; 
    //}
    //else { //diffuse only

    //    glm::vec3 ray_dir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    //    pathSegment.ray.origin = intersect + EPSILON * normal;

    //    pathSegment.ray.direction = ray_dir;
    //    pathSegment.color *= m.color;

    //}

}

