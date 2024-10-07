#include "interactions.h"

// Generate a random unit vector using spherical coordinates
__host__ __device__ glm::vec3 random_unit_vector(thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u02(0, 2);
    thrust::uniform_real_distribution<float> u0twoPi(0, TWO_PI);

    // Generate random angles
    float z = u02(rng) - 1.0f;  // Random float in [-1, 1] (cosine of the polar angle)
    float phi = u0twoPi(rng);   // Random azimuthal angle in [0, 2*pi]

    // Convert spherical coordinates to Cartesian coordinates
    float r = sqrt(1.0f - z * z);  // Radius in xy-plane (since x^2 + y^2 + z^2 = 1)
    float x = r * cos(phi);
    float y = r * sin(phi);

    return glm::vec3(x, y, z);
}

__host__ __device__ float schlick_reflectance(const float& cosine, const float& refraction_index)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    const glm::vec3& normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng));          // cos(theta) - taking the square root to ensure that the directions closer to the surface normal are favored and we dont get too many samples near the edges of the hemisphere
    float over = sqrt(1.0f - up * up);     // sin(theta) = sqrt(1 - cos(theta) * cos(theta))
    float around = u01(rng) * TWO_PI;   // angle in the plane around the normal

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    //
    // Why do this?
    // We need two perpendicular vectors to the normal to generate a random direction in the hemisphere
    // This avoids degenerate cases where the cross product would be zero if the normal and the chosen vector were parallel

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
    PathSegment& pathSegment,
    const glm::vec3& intersect_point,
    const glm::vec3& normal,
    const bool& front_face,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 scatter_direction;
    glm::vec3 color;

    if (m.hasReflective && m.hasRefractive) // dielectric
    {
        float ri = front_face ? (1.0f / m.indexOfRefraction) : m.indexOfRefraction;
        float cos_theta = min(glm::dot(-pathSegment.ray.direction, normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        thrust::uniform_real_distribution<float> u01(0, 1);

        if (cannot_refract || schlick_reflectance(cos_theta, ri) > u01(rng))
        {
            scatter_direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else
        {
            scatter_direction = glm::refract(pathSegment.ray.direction, normal, ri);
        }

        color = m.color;
    }
    else if (m.hasReflective) // metal
    {
        // We can also randomize the reflected direction by using a small sphere and choosing a new endpoint for the ray. We'll
        // use a random point from the surface of a sphere centered on the original endpoint, scaled by the roughness factor.
        // The bigger the roughness sphere, the rougher the reflections will be. This suggests adding a roughness parameter that is just
        // the radius of the sphere (so zero is no perturbation). The catch is that for big spheres or grazing rays, we may scatter
        // below the surface. We can just have the surface absorb those.
        // Also note that in order for the roughness sphere to make sense, it needs to be consistently scaled compared to the reflection
        // vector. To address this, we need to normalize the reflected ray.

        // both arguments are already normalized, so the reflected ray will be normalized
        scatter_direction = glm::reflect(pathSegment.ray.direction, normal);

        // my dirty, relatively cheap way of capturing the complexity of the fresnel effect for metals
        // this simulates an exponential dropoff of roughness as we reach grazing angles
        // cos_theta = 0:       fresnel_adjusted_roughness_factor = 0.0
        // cos_theta = 0.25:    fresnel_adjusted_roughness_factor = 0.354
        // cos_theta = 0.5:     fresnel_adjusted_roughness_factor = 0.707
        // cos_theta = 0.75:    fresnel_adjusted_roughness_factor = 0.931
        // cos_theta = 1:       fresnel_adjusted_roughness_factor = 1.0
        float cos_theta = min(glm::dot(-pathSegment.ray.direction, normal), 1.0f);
        float fresnel_adjusted_roughness_factor = pow(cos_theta, 1.0f - cos_theta);

        // roughness already clamped to 0, 1
        scatter_direction += fresnel_adjusted_roughness_factor * m.roughness * random_unit_vector(rng);

        // The catch is that for big spheres or grazing rays, we may scatter below the surface. We can just have the surface absorb those
        if (glm::dot(scatter_direction, normal) > 0.0f)
        {
            color = m.color;
        }
        else
        {
            color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
    //else if (m.hasRefractive)
    //{

    //}
    else // Lambertian Diffuse
    {
        scatter_direction = calculateRandomDirectionInHemisphere(normal, rng);

        // No need to scale the reflected color by abs(cos(theta))
        // since calculateRandomDirectionInHemisphere biases the sample directions toward the normal of the surface by sampling the hemisphere cosine-weighted.
        // This implicitly incorporates the cosine term into the light scattering model.
        color = m.color;
    }

    pathSegment.ray.direction = glm::normalize(scatter_direction);
    pathSegment.ray.origin = intersect_point;
    pathSegment.color *= color;
}
