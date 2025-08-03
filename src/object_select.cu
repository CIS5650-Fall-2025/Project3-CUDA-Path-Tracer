#include "object_select.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


SelectionState selection;


__global__ void findClosestHitKernel(
    Camera cam,
    Geom* geoms,
    int numGeoms,
    glm::vec2 screenPixel,
    ShadeableIntersection* outIntersection,
    int* outGeomID)
{
    // only one thread performs the pick
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // build pick ray 
    float px = ((float)screenPixel.x + 0.5f - cam.resolution.x * 0.5f) * cam.pixelLength.x;
    float py = ((float)screenPixel.y + 0.5f - cam.resolution.y * 0.5f) * cam.pixelLength.y;

    glm::vec3 rayDir = glm::normalize(
        cam.view
        + cam.right * px
        - cam.up * py
    );

    Ray ray;
    ray.origin = cam.position;
    ray.direction = rayDir;

    // debug NDC if you still want (optional)
    glm::vec2 ndc = glm::vec2(
        (screenPixel.x / cam.resolution.x) * 2.0f - 1.0f,
        (screenPixel.y / cam.resolution.y) * 2.0f - 1.0f
    );
    ndc.y = -ndc.y;

    int hitID = -1;
    ShadeableIntersection closest = {};
    closest.t = FLT_MAX;

    glm::vec3 out_intersect;
    glm::vec3 out_normal;
    glm::vec2 out_uv;
    bool outside;

    for (int i = 0; i < numGeoms; ++i) {
        const Geom& geom = geoms[i];

        // skip highlight shell itself
        if (geom.isHighlightShell) continue;

        float t = -1.0f;

        if (geom.type == CUBE) {
            t = boxIntersectionTest(geom, ray, out_intersect, out_normal, outside);
        }
        else if (geom.type == SPHERE) {
            t = sphereIntersectionTest(geom, ray, out_intersect, out_normal, outside);
        }
        else if (geom.type == MESH) {
            t = meshIntersectionTest_WithMeshBVH(geom, ray, out_intersect, out_normal, out_uv, outside);
        }

        if (t > 1e-4f && t < closest.t) {
            hitID = i;
            closest.t = t;
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



int pickObject(int mouseX, int mouseY, Scene* scene, Geom* dev_geoms) {
    // Prepare device buffers
    ShadeableIntersection* dIsect;
    int* dHitID;
    cudaMalloc(&dIsect, sizeof(ShadeableIntersection));
    cudaMalloc(&dHitID, sizeof(int));

    // Launch 1x1 kernel
    dim3 threads(1), blocks(1);
    findClosestHitKernel <<<blocks, threads>>> (
        scene->state.camera,
        dev_geoms,
        (int)scene->geoms.size(),
        glm::vec2((float)mouseX, (float)mouseY),
        dIsect,
        dHitID
        );
    cudaDeviceSynchronize();

    //Copy back
    int  hHitID;
    cudaMemcpy(&hHitID, dHitID, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(dIsect);
    cudaFree(dHitID);

    return hHitID;
}




void addHighlightShell(int pickedID, Scene* scene)
{
    // Clean up any existing shell (whether flag was correct or not)
    if (!scene->geoms.empty()) {
        Geom& last = scene->geoms.back();
        if (last.isHighlightShell) {
            scene->geoms.pop_back();
        }
    }

    if (pickedID < 0 || pickedID >= scene->geoms.size()) return;

    const Geom& base = scene->geoms[pickedID];
    Geom shell = base;

    shell.materialid = 0; //0 is default material id preserved for highlight mat

    shell.scale = base.scale * (1.0f + selection.outlineThickness);
    //shell.translation += glm::vec3(0.0f, 10.0f, 0.0f); // upward shift

    //printf("Base scale: (%.2f, %.2f, %.2f)\n", base.scale.x, base.scale.y, base.scale.z);
    //printf("Shell scale: (%.2f, %.2f, %.2f)\n", shell.scale.x, shell.scale.y, shell.scale.z);


    shell.transform = utilityCore::buildTransformationMatrix(
        shell.translation, shell.rotation, shell.scale);
    shell.inverseTransform = glm::inverse(shell.transform);
    shell.invTranspose = glm::transpose(shell.inverseTransform);
    shell.isHighlightShell = true;

    scene->geoms.push_back(shell);
}


void removeHighlightShell(Scene* scene) {
    // Always attempt to remove trailing shell if present
    if (scene->geoms.empty()) {
        return;
    }

    Geom& last = scene->geoms.back();
    if (!last.isHighlightShell) {
        return;
    }

    scene->geoms.pop_back();
}
