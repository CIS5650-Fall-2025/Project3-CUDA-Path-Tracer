#include "object_select.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void findClosestHitKernel(
    Camera cam,
    Geom* geoms,
    int numGeoms,
    glm::vec2 screenPixel,
    ShadeableIntersection* outIntersection,
    int* outGeomID)
{

    // Generate ray direction from pixel
    glm::vec2 ndc = glm::vec2(
        (screenPixel.x / cam.resolution.x) * 2.0f - 1.0f,
        (screenPixel.y / cam.resolution.y) * 2.0f - 1.0f
    );
    ndc.y = -ndc.y; // Flip Y

    glm::vec3 rayDir = glm::normalize(
        cam.view +
        ndc.x * cam.pixelLength.x * cam.right +
        ndc.y * cam.pixelLength.y * cam.up
    );

    Ray ray;
    ray.origin = cam.position;
    ray.direction = rayDir;


    int hitID = -1;

    ShadeableIntersection closest = {};
    closest.t = 1e20f;


    glm::vec3 out_intersect;
    glm::vec3 out_normal;
    glm::vec2 out_uv;
    bool outside;

    for (int i = 0; i < numGeoms; ++i) {
        Geom geom = geoms[i];

        float t = 0;
        glm::vec3 n;
        glm::vec3 intersectPt;

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, ray, out_intersect, out_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, ray, out_intersect, out_normal, outside);
        }
        else if (geom.type == MESH) {
            t = meshIntersectionTest_WithMeshBVH(geom, ray, out_intersect, out_normal, out_uv, outside);

        }

        if (t > EPSILON && t < closest.t) {
            hitID = i;
            closest.t = t;
            //world coord
            closest.surfaceNormal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(out_normal, 0.0f)));
            closest.uv = out_uv;
            closest.intersectionPoint = out_intersect;
            closest.materialId = geom.materialid;
        }


    }

    if (hitID != -1) {
        *outIntersection = closest;
        *outGeomID = hitID;
    }
    else {
        *outGeomID = -1;

        outIntersection->t = -1.0f;
        outIntersection->surfaceNormal = glm::vec3(0.0f);
        outIntersection->uv = glm::vec2(0.0f);
        outIntersection->intersectionPoint = glm::vec3(0.0f);
        outIntersection->materialId = -1;
    }
}


void addHighlightShell(int pickedID, Scene* scene)
{
    if (selection.isShellActive) return;

    if (pickedID < 0 || pickedID >= scene->geoms.size()) return;

    const Geom& base = scene->geoms[pickedID];
    Geom shell = base;

    shell.materialid = 0; //0 is default material id preserved for highlight mat

    shell.scale *= (1.0f + selection.outlineScale);
    shell.translation += glm::vec3(0.0f, 10.0f, 0.0f); // upward shift


    printf("Base scale: (%.2f, %.2f, %.2f)\n", base.scale.x, base.scale.y, base.scale.z);
    printf("Shell scale: (%.2f, %.2f, %.2f)\n", shell.scale.x, shell.scale.y, shell.scale.z);




    shell.transform = utilityCore::buildTransformationMatrix(
        shell.translation, shell.rotation, shell.scale);
    shell.inverseTransform = glm::inverse(shell.transform);
    shell.invTranspose = glm::transpose(shell.inverseTransform);

    shell.isHighlightShell = true;

    selection.isShellActive = true;

    scene->geoms.push_back(shell);
}


void removeHighlightShell(Scene* scene) {
    if (!selection.isShellActive) return;

    //the highlight shell was the most recently added geometry
    if (!scene->geoms.empty()) {
        scene->geoms.pop_back();
    }

    selection.isShellActive = false;
}
