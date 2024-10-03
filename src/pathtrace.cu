#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define STREAMCOMPACT

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
    thrust::default_random_engine
    makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter, glm::vec3 *image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene *hst_scene = NULL;
static GuiDataContainer *guiData = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Mesh *dev_meshes = NULL;
static Tri *dev_tris = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static ShadeableIntersection *dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer *imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene *scene)
{
    hst_scene = scene;

    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Tri));
    cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Tri), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_tris);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    checkCUDAError("pathtraceFree");
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment *pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        int index = x + (y * cam.resolution.x);
        PathSegment &segment = pathSegments[index];

        Ray ray;
        ray.origin = cam.position;
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.radiance = glm::vec3();

        auto rng = makeSeededRandomEngine(iter, index, -1);
        thrust::uniform_real_distribution<float> u01(0, 1);

        ray.direction = glm::normalize(cam.view - cam.right * cam.pixelLength.x * (static_cast<float>(x) + u01(rng) - static_cast<float>(cam.resolution.x) * 0.5f) - cam.up * cam.pixelLength.y * (static_cast<float>(y) + u01(rng) - static_cast<float>(cam.resolution.y) * 0.5f));

        if (cam.lensSize == 0)
        {
            segment.ray = ray;
        }
        else
        {
            glm::vec2 diskPoint = calculateRandomPointOnDisk(rng) * cam.pixelLength * glm::vec2(cam.resolution);
            segment.ray.origin = cam.position + cam.lensSize * (diskPoint.x * cam.right + diskPoint.y * cam.up);

            glm::vec3 focalPoint = cam.position + cam.focalDist * ray.direction;
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        }
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment *pathSegments,
    Geom *geoms,
    int geoms_size,
    Tri *tris,
    int tris_size,
    ShadeableIntersection *intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths)
    {
        return;
    }

    intersections[path_index] = queryIntersection(pathSegments[path_index].ray, geoms, geoms_size, tris, tris_size);
}

__global__ void chooseLights(
    int numLights,
    int numPaths,
    int iter,
    int *lightIndices)
{
    size_t idx = blockIdx.x * blockDim.x * threadIdx.x;
    if (idx >= numPaths)
    {
        return;
    }
    auto rng = makeSeededRandomEngine(iter, idx, 0);
    thrust::uniform_int_distribution<int> dist(0, numLights - 1);
    lightIndices[idx] = dist(rng);
}

__global__ void shadeMaterialDirect(
    int num_paths,
    int iter,
    int lightCount,
    ShadeableIntersection *shadeableIntersections,
    Geom *geoms,
    int geomsSize,
    Tri *tris,
    int trisSize,
    PathSegment *pathSegments,
    Material *materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment &segment = pathSegments[idx];

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t <= 0)
    {
        segment.remainingBounces = 0;
        return;
    }

    const Material &material = materials[intersection.materialId];
    if (material.emittance > 0.f)
    {
        segment.radiance = segment.throughput * material.color * material.emittance;
        return;
    }

    glm::vec3 viewPoint = getPointOnRay(segment.ray, intersection.t);

    auto rng = makeSeededRandomEngine(iter, idx, 0);
    thrust::uniform_int_distribution<int> dist(0, lightCount - 1);
    int lightIndex = dist(rng);

    Sample lightSample = sampleLight(viewPoint, geoms[lightIndex], materials, rng);
    Ray checkRay{.origin = viewPoint + EPSILON * lightSample.incomingDirection, .direction = lightSample.incomingDirection};
    int shadowResult = queryIntersectionGeometryIndex(checkRay, geoms, geomsSize, tris, trisSize);

    if (shadowResult != lightIndex)
    {
        lightSample.value = glm::vec3(0, 0, 0);
    }
    segment.throughput = getBsdf(materials[intersection.materialId], intersection.surfaceNormal, lightSample.incomingDirection, segment.ray.direction);
    segment.radiance = segment.throughput * lightSample.value / lightSample.pdf;
}

// The actual entrypoint for shading a material
__global__ void shadeMaterialSimple(
    int iter,
    int num_paths,
    ShadeableIntersection *shadeableIntersections,
    Geom *geoms,
    PathSegment *pathSegments,
    Material *materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment &segment = pathSegments[idx];

#ifndef STREAMCOMPACT
    if (segment.remainingBounces <= 0)
    {
        return;
    }
#endif

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t <= 0)
    {
        segment.radiance = glm::vec3();
        segment.remainingBounces = 0;
        return;
    }

    const Material &material = materials[intersection.materialId];
    if (material.emittance > 0.f)
    {
        segment.radiance = segment.throughput * material.color * material.emittance;
        segment.remainingBounces = 0;
        return;
    }

    glm::vec3 intersect = getPointOnRay(segment.ray, intersection.t);
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);
    scatterRay(segment, intersect, intersection.surfaceNormal, material, rng);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image, PathSegment *iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.radiance;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment *dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    int active_paths = num_paths;

    if (hst_scene->useDirectLighting)
    {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (active_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
            depth,
            active_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_tris,
            hst_scene->tris.size(),
            dev_intersections);
        cudaDeviceSynchronize();
        depth++;
        shadeMaterialDirect<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            iter,
            hst_scene->numLights,
            dev_intersections,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_tris,
            hst_scene->tris.size(),
            dev_paths,
            dev_materials);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
    else
    {
        while (!iterationComplete)
        {
            // clean shading chunks
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

            // tracing
            dim3 numblocksPathSegmentTracing = (active_paths + blockSize1d - 1) / blockSize1d;
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth,
                active_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_tris,
                hst_scene->tris.size(),
                dev_intersections);
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            depth++;

            shadeMaterialSimple<<<numblocksPathSegmentTracing, blockSize1d>>>(
                iter,
                active_paths,
                dev_intersections,
                dev_geoms,
                dev_paths,
                dev_materials);

#ifdef STREAMCOMPACT
            active_paths = thrust::partition(thrust::device, dev_paths, dev_paths + active_paths, PathActive()) - dev_paths;
            iterationComplete = active_paths == 0 || depth > traceDepth;
#endif
#ifndef STREAMCOMPACT
            iterationComplete = depth > traceDepth;
#endif

            if (guiData != NULL)
            {
                guiData->TracedDepth = depth;
            }
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
