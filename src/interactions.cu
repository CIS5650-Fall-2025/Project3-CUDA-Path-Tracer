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

// implement the kernel function that scatters rays based on material types
__host__ __device__ void scatter(const glm::vec3 intersection_color,
                                 const glm::vec3 intersection_point,
                                 const glm::vec3 intersection_normal,
                                 const Material intersection_material,
                                 thrust::default_random_engine generator,
                                 PathSegment& pathSegment) {

    // set the new ray's origin to somewhere slightly above the intersection point
    pathSegment.ray.origin = intersection_point + intersection_normal * EPSILON;

    // handle the mirror material
    if (intersection_material.hasReflective == 1.0f) {

        // reflect the ray's direction
        pathSegment.ray.direction = glm::reflect(
            pathSegment.ray.direction, intersection_normal
        );

        // accumulate the output color
        pathSegment.color *= intersection_color;

        // handle the reflective material
    } else if (intersection_material.hasReflective > 0.0f) {

        // compute the reflection direction
        const glm::vec3 reflection {glm::reflect(
            pathSegment.ray.direction, intersection_normal
        )};

        // compute the direction of the hemisphere to shoot the random rays from
        const glm::vec3 direction {glm::normalize(
            glm::mix(intersection_normal, reflection, intersection_material.hasReflective)
        )};

        // set the new ray's direction to a random direction
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(
            direction, generator
        );

        // shift the new ray's direction towards the hemisphere's direction based on reflectivity
        pathSegment.ray.direction = glm::normalize(
            glm::mix(pathSegment.ray.direction, direction, intersection_material.hasReflective)
        );

        // accumulate the output color
        pathSegment.color *= intersection_color;

        // handle the purely refractive material
    } else if (intersection_material.hasRefractive == 1.0f) {

        // acquire the direction of the ray
        const glm::vec3 direction {glm::normalize(pathSegment.ray.direction)};

        // acquire the intersection normal
        glm::vec3 normal {glm::normalize(intersection_normal)};

        // generate a random decimal
        thrust::uniform_real_distribution<float> distribution (0.0f, 1.0f);
        const float random_decimal {distribution(generator)};

        // compute the Fresnel factor using the Schlick's approximation
        const float cosine {-glm::dot(direction, normal)};
        const float n_1 {1.0f};
        const float n_2 {intersection_material.indexOfRefraction};
        const float R_0 {glm::pow((n_1 - n_2) / (n_1 + n_2), 2.0f)};
        const float factor {R_0 + (1.0f - R_0) * glm::pow(1.0f - cosine, 5.0f)};

        // perform refraction when the Fresnel factor is small
        if (random_decimal > factor) {

            // compute the refraction ratio based on the material's index of refraction
            float ratio {1.0f / intersection_material.indexOfRefraction};

            // determine whether the ray exiting from the surface
            if (glm::dot(direction, normal) >= 0.0f) {

                // update the normal
                normal = -normal;

                // update the refraction ratio
                ratio = intersection_material.indexOfRefraction;
            }

            // compute the refracted ray direction
            pathSegment.ray.direction = glm::refract(direction, normal, ratio);

            // set the new ray's origin
            pathSegment.ray.origin = intersection_point + pathSegment.ray.direction * 0.01f;

            // reflect the ray's direction when the Fresnel factor is big
        } else {
            pathSegment.ray.direction = glm::reflect(
                pathSegment.ray.direction, intersection_normal
            );
        }

        // accumulate the output color
        pathSegment.color *= intersection_color;

        // handle the diffuse material
    } else {

        // set the new ray's direction to a random direction in the hemisphere
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(
            intersection_normal, generator
        );

        // accumulate the output color
        pathSegment.color *= intersection_color;
    }

    // decrease the number of remaining bounces
    pathSegment.remainingBounces -= 1;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // set the new ray's origin to somewhere slightly above the intersection point
    pathSegment.ray.origin = intersect + normal * EPSILON;

    // handle the mirror material
    if (m.hasReflective == 1.0f) {

        // reflect the ray's direction
        pathSegment.ray.direction = glm::reflect(
            pathSegment.ray.direction, normal
        );

        // accumulate the output color
        pathSegment.color *= m.color;

        // handle the reflective material
    } else if (m.hasReflective > 0.0f) {

        // compute the reflection direction
        const glm::vec3 reflection {glm::reflect(
            pathSegment.ray.direction, normal
        )};

        // compute the direction of the hemisphere to shoot the random rays from
        const glm::vec3 direction {glm::normalize(
            glm::mix(normal, reflection, m.hasReflective)
        )};

        // set the new ray's direction to a random direction
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(
            direction, rng
        );

        // shift the new ray's direction towards the hemisphere's direction based on reflectivity
        pathSegment.ray.direction = glm::normalize(
            glm::mix(pathSegment.ray.direction, direction, m.hasReflective)
        );

        // accumulate the output color
        pathSegment.color *= m.color;

        // handle the diffuse material
    } else {

        // set the new ray's direction to a random direction in the hemisphere
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(
            normal, rng
        );

        // accumulate the output color
        pathSegment.color *= m.color;
    }

    // decrease the number of remaining bounces
    pathSegment.remainingBounces -= 1;
}
