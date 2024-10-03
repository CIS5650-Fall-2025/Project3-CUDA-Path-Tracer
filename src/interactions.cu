#include "interactions.h"

__host__ __device__ void swap(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal,
                                                                   thrust::default_random_engine &rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up = sqrt(u01(rng));       // cos(theta)
  float over = sqrt(1 - up * up);  // sin(theta)
  float around = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(1, 0, 0);
  } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(0, 1, 0);
  } else {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal + cos(around) * over * perpendicularDirection1 + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(PathSegment &pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material &m,
                                    thrust::default_random_engine &rng) {
  // A basic implementation of pure-diffuse shading will just call the
  // calculateRandomDirectionInHemisphere defined above.

  if (m.emittance > 0.0f) {
    pathSegment.remainingBounces = 0;
    return;
  }

  thrust::uniform_real_distribution<float> u01(0, 1);

  if (m.hasRefractive > 0.0f) {
    // If refractive
    float etaI = 1.0f;
    float etaT = m.indexOfRefraction;

    glm::vec3 incidentDir = glm::normalize(pathSegment.ray.direction);
    glm::vec3 normalDir = glm::normalize(normal);
    float cosThetaI = glm::dot(-incidentDir, normalDir);
    bool entering = cosThetaI > 0.0f;

    if (!entering) {
      swap(etaI, etaT);
      normalDir = -normalDir;
      cosThetaI = glm::dot(-incidentDir, normalDir);
    }

    float eta = etaI / etaT;
    float sinThetaTSquared = eta * eta * (1.0f - cosThetaI * cosThetaI);

    if (sinThetaTSquared > 1.0f) {
      // Total internal reflection
      glm::vec3 reflectedDir = glm::reflect(incidentDir, normalDir);
      pathSegment.ray.origin = intersect + 0.001f * normalDir;
      pathSegment.ray.direction = reflectedDir;
      pathSegment.color *= m.specular.color;
    } else {
      // Fresnel reflectance using Schlick's approximation
      float cosThetaT = sqrt(1.0f - sinThetaTSquared);
      float R0 = pow((etaI - etaT) / (etaI + etaT), 2.0f);
      float R = R0 + (1.0f - R0) * pow(1.0f - cosThetaI, 5.0f);

      float rand = u01(rng);
      if (rand < R) {
        // Reflect
        glm::vec3 reflectedDir = glm::reflect(incidentDir, normalDir);
        pathSegment.ray.origin = intersect + 0.001f * normalDir;
        pathSegment.ray.direction = reflectedDir;
        pathSegment.color *= m.specular.color / R;
      } else {
        // Refract
        glm::vec3 refractedDir = eta * incidentDir + (eta * cosThetaI - cosThetaT) * normalDir;
        pathSegment.ray.origin = intersect - 0.001f * normalDir;
        pathSegment.ray.direction = glm::normalize(refractedDir);
        pathSegment.color *= m.specular.color / (1.0f - R);
      }
    }
  } else if (m.hasReflective > 0.0f) {
    // If reflective
    glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.ray.origin = intersect + 0.001f * normal;
    pathSegment.ray.direction = reflectedDir;
    pathSegment.color *= m.specular.color;
  } else {
    // Diffuse and specular materials
    float diffuse_intensity = glm::length(m.color);
    float specular_intensity = glm::length(m.specular.color);
    float total_intensity = diffuse_intensity + specular_intensity;

    if (total_intensity == 0.0f) {
      pathSegment.remainingBounces = 0;
      return;
    }

    float prob_diffuse = diffuse_intensity / total_intensity;
    float prob_specular = specular_intensity / total_intensity;
    float rand = u01(rng);

    if (rand < prob_diffuse) {
      // Diffuse reflection
      glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);
      pathSegment.ray.origin = intersect + 0.001f * normal;
      pathSegment.ray.direction = glm::normalize(newDir);
      pathSegment.color *= m.color / prob_diffuse;
    } else {
      // Specular reflection
      glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
      pathSegment.ray.origin = intersect + 0.001f * normal;
      pathSegment.ray.direction = glm::normalize(reflectedDir);
      pathSegment.color *= m.specular.color / prob_specular;
    }
  }
}
