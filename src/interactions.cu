#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
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



__host__ __device__ void buildTangentSpace(const glm::vec3& n, glm::vec3& T, glm::vec3& B) {
    if (fabs(n.z) < 0.999f) {
        T = glm::normalize(glm::cross(n, glm::vec3(0.0f, 0.0f, 1.0f)));
    }
    else {
        T = glm::normalize(glm::cross(n, glm::vec3(0.0f, 1.0f, 0.0f)));
    }
    B = glm::cross(T, n);
}


__host__ __device__ glm::vec3 sampleGGX(glm::vec3 normal, float roughness, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float alpha = roughness * roughness;
    float u1 = u01(rng);
    float u2 = u01(rng);

    float theta = atanf(alpha * sqrtf(u1) / sqrtf(1.0f - u1));
    float phi = 2.0f * PI * u2;

    float sinTheta = sinf(theta);
    float cosTheta = cosf(theta);

    // Convert spherical to cartesian
    glm::vec3 h_local(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta
    );

    // Transform to world space
    glm::vec3 T, B;
    buildTangentSpace(normal, T, B);
    glm::vec3 h_world = h_local.x * T + h_local.y * B + h_local.z * normal;

    return glm::normalize(h_world);
}


__host__ __device__ void accumulateFirstBounce(
    int index,
    glm::vec3 normal,
    glm::vec3 color,
    glm::vec3* normals,
    glm::vec3* albedo)
{
    normals[index] += normal;
    albedo[index] += color;
}




//This function implements diffuse BRDF sampling.
// - It samples a new direction for the ray according to Lambertian reflection.
// - It updates the ray direction and its color(throughput) accordingly.
// - This function is called when the material is purely diffuse.
__host__ __device__ void sample_f_diffuse(
    PathPayload& payload, thrust::default_random_engine& rng)
{
    PathSegment& pathSegment = *payload.path;
    glm::vec3 normal = glm::normalize(payload.intersection.surfaceNormal);
    const Material& m = payload.material;

    glm::vec3 rand_dir = calculateRandomDirectionInHemisphere(normal, rng);

    pathSegment.ray.direction = glm::normalize(rand_dir);
    pathSegment.color *= m.color;

}



__host__ __device__ void sample_f_specular(
    PathPayload& payload, thrust::default_random_engine& rng) {

    PathSegment& pathSegment = *payload.path;
    glm::vec3 normal = glm::normalize(payload.intersection.surfaceNormal);
    const Material& m = payload.material;

    glm::vec3 incident = glm::normalize(pathSegment.ray.direction);

    glm::vec3 reflect_dir = glm::reflect(incident, normal);
    pathSegment.ray.direction = reflect_dir;

    // Glossy specular (Rough metals, rough glass, polished surfaces, etc)
    if (m.roughness > 0.0f) {
        glm::vec3 rand_dir = calculateRandomDirectionInHemisphere(normal, rng);

        glm::vec3 new_dir = glm::normalize(glm::mix(reflect_dir, rand_dir, m.roughness));
        pathSegment.ray.direction = new_dir;

        //glm::vec3 h = sampleGGX(normal, m.roughness, rng);
        //glm::vec3 reflected = glm::reflect(incident, h);
        //pathSegment.ray.direction = glm::normalize(reflected);
    }

    pathSegment.color *= m.specular.color;
}



__host__ __device__ void sample_f_dielectric(
    PathPayload& payload, thrust::default_random_engine& rng)
{
    PathSegment& pathSegment = *payload.path;
    glm::vec3 normal = glm::normalize(payload.intersection.surfaceNormal);
    const Material& m = payload.material;

    glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
    normal = glm::normalize(normal);

    //refractive index of the medium the ray is coming from.
    //Defaulted to 1.0 for air.
    float etaA = 1.0f;
    float etaB = m.indexOfRefraction;
    float cosTheta_i = -glm::dot(incident, normal);

    //handle flipping
    bool entering = (cosTheta_i > 0.0f);
    float eta_i = entering ? etaA : etaB;
    float eta_t = entering ? etaB : etaA;
    normal = entering ? normal : -normal;
    cosTheta_i = entering ? cosTheta_i : -cosTheta_i;

    float eta = eta_i / eta_t;

    // Total internal reflection test BEFORE calling glm::refract
    float sin2Theta_t = eta * eta * (1.0f - cosTheta_i * cosTheta_i);
    bool totalInternalReflection = (sin2Theta_t > 1.0f);

    // Schlick approximation for Fresnel
    float R0 = (eta_i - eta_t) / (eta_i + eta_t);
    R0 = R0 * R0;
    float fresnelReflectance = R0 + (1.0f - R0) * powf(1.0f - cosTheta_i, 5.0f);

    // Sample reflection or refraction
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float rand = u01(rng);

    if (totalInternalReflection || rand < fresnelReflectance) {
        // Reflect
        glm::vec3 new_dir = glm::reflect(incident, normal);
        pathSegment.ray.direction = glm::normalize(new_dir);
        pathSegment.color *= m.specular.color;

        payload.recordFirstBounce(normal, m.specular.color);

    }
    else {
        // Refract
        glm::vec3 new_dir = glm::refract(incident, normal, eta);
        pathSegment.ray.direction = glm::normalize(new_dir);
        pathSegment.color *= m.color;
    }
}






__device__ void scatterRay(PathPayload& payload, thrust::default_random_engine& rng) {
    PathSegment& path = *payload.path;
    glm::vec3 normal = glm::normalize(payload.intersection.surfaceNormal);
    glm::vec3 hitPoint = getPointOnRay(path.ray, payload.intersection.t);

    path.ray.origin = hitPoint;

    const Material& m = payload.material;

    if (m.emittance > 0.0f) {
        path.color *= (m.color * m.emittance);
        path.remainingBounces = 0;

        payload.recordFirstBounce(glm::vec3(0.0f), m.color * m.emittance);

        return;
    }
    else {
        if (m.hasRefractive > 0.0f) {
            sample_f_dielectric(payload, rng);
        }
        else if (m.hasReflective > 0.0f) {
            sample_f_specular(payload, rng);

            payload.recordFirstBounce(normal, m.specular.color);
        }
        else {
            sample_f_diffuse(payload, rng);
            payload.recordFirstBounce(normal, m.color);

        }

        path.remainingBounces--;
    }

    path.ray.origin += path.ray.direction * 0.01f;

}
