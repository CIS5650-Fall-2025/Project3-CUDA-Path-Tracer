#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
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

struct path_has_remaining_bounces
{
    __device__ bool operator()(const PathSegment& pathSegment) const {
        return pathSegment.remainingBounces > 0;
    }
};

struct compare_intersections_by_materialId
{
    __device__ bool operator()(const ShadeableIntersection& isect0, const ShadeableIntersection& insect1) const {
        return isect0.materialId < insect1.materialId;
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(const int& iter, const int& index, const int& depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
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

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static BVHNode* dev_bvh_nodes = NULL;
static int* dev_bvh_geom_idx = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_bvh_nodes, scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvh_nodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvh_geom_idx, scene->geomIdx.size() * sizeof(int));
    cudaMemcpy(dev_bvh_geom_idx, scene->geomIdx.data(), scene->geomIdx.size() * sizeof(int), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_bvh_nodes);
    cudaFree(dev_bvh_geom_idx);

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
__global__ void generateRayFromCamera(const Camera cam, const bool stochasticSampling, const int iter, const int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        float sampled_x = (float)x;
        float sampled_y = (float)y;
        if (stochasticSampling)
        {
            // Implement antialiasing by jittering the ray
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            // Random offset in the [-.5, .5] range for x and y
            // Apply stochastic sampling (jittered ray direction)
            sampled_x += u01(rng) - 0.5f;
            sampled_y += u01(rng) - 0.5f;
        }

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampled_x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampled_y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    const int num_paths,
    const PathSegment* pathSegments,
    const Geom* geoms,
    const int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        const PathSegment& pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        bool outside;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool tmp_outside;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == TRIANGLE) {

                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > RAY_TRACE_EPSILON && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }

        ShadeableIntersection& intersection = intersections[path_index];
        if (hit_geom_index == -1)
        {
            intersection.t = -1.0f;
            intersection.materialId = geoms_size;
        }
        else
        {
            // The ray hits something
            intersection.t = t_min;
            intersection.materialId = geoms[hit_geom_index].materialid;
            intersection.surfaceNormal = normal;
            intersection.front_face = outside;
        }
    }
}

__global__ void computeIntersectionsBVH(
    const int num_paths,
    const PathSegment* pathSegments,
    const Geom* geoms,
    const int geoms_size,
    const BVHNode* bvhNodes,
    const int bvhRootIndex,
    const int* geomIndices,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        const PathSegment& pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        bool outside;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool tmp_outside;

        // Initialize a stack for BVH traversal
        int stack[64];
        int stackSize = 0;
        stack[stackSize++] = bvhRootIndex;  // Start with the root node

        while (stackSize > 0)
        {
            int currentNodeIndex = stack[--stackSize];
            const BVHNode& node = bvhNodes[currentNodeIndex];

            // Check if ray intersects the bounding box of this node
            if (aabbIntersectionTest(node.aabb, pathSegment.ray))
            {
                if (node.geomCount > 0) // if leaf node
                {
                    for (int i = 0; i < node.geomCount; i++)
                    {
                        int geomIdx = geomIndices[node.leftFirst + i];
                        const Geom& geom = geoms[geomIdx];

                        if (geom.type == CUBE)
                        {
                            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                        }
                        else if (geom.type == SPHERE)
                        {
                            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                        }
                        else if (geom.type == TRIANGLE) {

                            t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                        }

                        // Compute the minimum t from the intersection tests to determine what
                        // scene geometry object was hit first.
                        if (t > RAY_TRACE_EPSILON && t_min > t)
                        {
                            t_min = t;
                            hit_geom_index = geomIdx;
                            intersect_point = tmp_intersect;
                            normal = tmp_normal;
                            outside = tmp_outside;
                        }
                    }
                }
                else
                {
                    // Push child nodes onto the stack for traversal
                    stack[stackSize++] = node.leftFirst + 1;
                    stack[stackSize++] = node.leftFirst;
                }
            }
        }

        ShadeableIntersection& intersection = intersections[path_index];
        if (hit_geom_index == -1)
        {
            intersection.t = -1.0f;
            intersection.materialId = geoms_size;
        }
        else
        {
            // The ray hits something
            intersection.t = t_min;
            intersection.materialId = geoms[hit_geom_index].materialid;
            intersection.surfaceNormal = normal;
            intersection.front_face = outside;
        }
    }
}

__global__ void shadeBSDF(
    const int iter,
    const int currentDepth,
    const int num_paths,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    const Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        const ShadeableIntersection& intersection = shadeableIntersections[idx];
        PathSegment& pathSegment = pathSegments[idx];

        if (intersection.t > 0.0f) // if intersection
        {
            const Material& material = materials[intersection.materialId];

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegment.color *= (material.color * material.emittance);
                pathSegment.remainingBounces = 0;
            }
            else {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, currentDepth);

                const glm::vec3 intersect_point = getPointOnRay(pathSegment.ray, intersection.t);
                scatterRay(pathSegment, intersect_point, intersection.surfaceNormal, intersection.front_face, material, rng);

                --pathSegment.remainingBounces;
            }
        }
        else // if no intersection
        {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(const int nPaths, glm::vec3* image, const PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        const PathSegment& iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize_2d(8, 8);
    const dim3 numBlocksPixels_2d(
        (cam.resolution.x + blockSize_2d.x - 1) / blockSize_2d.x,
        (cam.resolution.y + blockSize_2d.y - 1) / blockSize_2d.y);

    // 1D block for path tracing
    const int blockSize_1d = 128;
    dim3 numBlocksPixels_1d = (pixelcount + blockSize_1d - 1) / blockSize_1d;

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

    generateRayFromCamera<<<numBlocksPixels_2d, blockSize_2d>>>(cam, guiData->CameraRaysStochasticSampling, iter, traceDepth, dev_paths);
    checkCUDAError("generateRayFromCamera failed");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
        checkCUDAError("cudaMemset remaining dev_intersections to 0 failed!");

        // tracing
        dim3 numBlocksPathSegments_1d = (num_paths + blockSize_1d - 1) / blockSize_1d;
        if (guiData->BVH)
        {
            computeIntersectionsBVH<<<numBlocksPathSegments_1d, blockSize_1d>>> (
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_bvh_nodes,
                hst_scene->bvhRootIndex,
                dev_bvh_geom_idx,
                dev_intersections
            );
            checkCUDAError("computeIntersectionsBVH failed");
        }
        else
        {
            computeIntersections<<<numBlocksPathSegments_1d, blockSize_1d>>> (
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
            );
            checkCUDAError("computeIntersections failed");
        }
        cudaDeviceSynchronize();
        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.

        // Compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.


        // materialId set to geoms_size on no intersection
        if (guiData->SortPathSegmentsByMaterial)
        {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_intersections_by_materialId());
            checkCUDAError("thrust::sort_by_key failed");
        }

        shadeBSDF<<<numBlocksPathSegments_1d, blockSize_1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        checkCUDAError("shadeBSDF failed");

        dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, path_has_remaining_bounces());
        checkCUDAError("thrust::stable_partition failed");

        num_paths = dev_path_end - dev_paths;

        iterationComplete = num_paths == 0;

        guiData->TracedDepth = depth;
    }

    // Assemble this iteration and apply it to the image
    finalGather<<<numBlocksPixels_1d, blockSize_1d>>>(pixelcount, dev_image, dev_paths);
    checkCUDAError("finalGather failed");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<numBlocksPixels_2d, blockSize_2d>>>(pbo, cam.resolution, iter, dev_image);
    checkCUDAError("sendImageToPBO failed");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy dev_image to scene failed!");
}
