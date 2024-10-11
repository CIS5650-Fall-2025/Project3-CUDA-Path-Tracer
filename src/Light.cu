#include "Light.h"

__device__ glm::vec3 Sample_Li(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, int num_Lights, const glm::vec3& view_point, const glm::vec3& normal, glm::vec3& wiW, float& pdf,
    int& chosenLightIdx, int& chosenLightID, LightType& chosenLightType,
    thrust::default_random_engine& rng)
{
    // Choose a random light 
    // Area light, point light, spot lights
    // For now, let's do JUST area lights

    int n = num_Lights; //TODO- pass this in
    thrust::uniform_real_distribution<float> u01(0, 1);
    int randomLightIdx = int(u01(rng) * n);
    chosenLightIdx = randomLightIdx;

    chosenLightID = areaLights[chosenLightIdx].ID; //useful for when there are other kinds of lights ... e.g. point light
    chosenLightType = AREALIGHT;
    return DirectSampleAreaLight(triangles, bvhNodes, areaLights, randomLightIdx, num_Lights, view_point, normal, wiW, pdf, rng);
}

__device__ glm::vec3 DirectSampleAreaLight(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, 
    int idx, int num_Lights, 
    const glm::vec3& view_point, const glm::vec3& normal, 
    glm::vec3& wiW, float& pdf,
    thrust::default_random_engine& rng)
{
    const AreaLight light = areaLights[idx];
    Ray shadowRay;

    if (light.shapeType == RECTANGLE) {

        glm::vec3 pos = glm::vec3(light.transform * glm::vec4(0, 0, 0, 1));
        glm::vec2 sideLen = { light.scale.x, light.scale.y };
        glm::vec3 nor = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(0, 0, 1, 0)));

        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec2 xi = { u01(rng), u01(rng) };
        xi = xi * 2.f - glm::vec2(1.); //[-1, 1]
        glm::vec3 tan, bit;
        coordinateSystem(nor, tan, bit);

        glm::vec3 sample = xi.x * tan * sideLen.x + xi.y * bit * sideLen.y + pos;

        shadowRay.origin = view_point;
        shadowRay.direction = normalize(sample - view_point);
        wiW = shadowRay.direction; //point to light!

        // Convert PDF from w/r/t surface area to w/r/t solid angle
        float r2 = dot(sample - view_point, sample - view_point);
         ////r*r / (cos(theta_w) * area)
        pdf = r2 / (abs(dot(shadowRay.direction, nor)) * (4 * sideLen.x * sideLen.y));

        //Intersect scene with shadowRay...
        ShadeableIntersection intr;
        intr.t = -1;
        intr.areaLightId = -1;
        bool hitLight = DirectLightIntersectTest(intr, shadowRay,
            triangles, bvhNodes,
            areaLights, num_Lights);

        if (hitLight) {
            return ((float)num_Lights) * areaLights[idx].Le * areaLights[idx].emittance;
        }
        else {
            return glm::vec3(0, 0, 0);
        }
    }
    else if (light.shapeType == SPHERE) {

    }
    wiW = glm::vec3(-1);
    pdf = 0;
    return glm::vec3(0,0,0);
}