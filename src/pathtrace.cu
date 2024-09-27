#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <stack>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "bvh.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define PATHTRACE_CONTIGUOUS_MATERIALID 1;

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

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
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
static BVH::Node* dev_nodes = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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

    cudaMalloc(&dev_nodes, scene->nodes.size() * sizeof(BVH::Node));
    cudaMemcpy(dev_nodes, scene->nodes.data(), scene->nodes.size() * sizeof(BVH::Node), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_nodes);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Antialiasing: jitter rays by [0,1] to generate uniformly random direction distribution per pixel
        glm::vec2 jitter = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);

        PathSegment& segment = pathSegments[index];

        // reminder: change origin by aperture size * shape sample for DoF
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitter[0] - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitter[1]  - (float)cam.resolution.y * 0.5f)
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
    int depth,
    int numPaths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geomsSize,
    BVH::Node* nodes,
    int nodesSize,
    int rootIdx,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= numPaths) return;

    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    
    // Early terminate if no intersection with the root node
    glm::vec2 times;
    if (bboxIntersectionTest(nodes[rootIdx].bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times) == -1) return;

    // BVH intersection hierarchy
    int nodeStack[1024];
    memset(nodeStack, 0, 1024);
    int nodeStackFinger = 0;

    nodeStack[nodeStackFinger] = rootIdx;
    nodeStackFinger++;

    while (nodeStackFinger > 0) {
        int currIdx = nodeStack[nodeStackFinger - 1];
        const BVH::Node& node = nodes[currIdx];
        nodeStackFinger--;

        bool hit = bboxIntersectionTest(node.bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times);
        if (!hit || hit && times[0] > t_min) continue;

        if (node.l == node.r) {
            for (int i = node.start; i < node.start + node.size; i++) {
                Geom& geom = geoms[i];
                if (geom.type == CUBE)
                {
                    t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }
                else if (geom.type == SPHERE)
                {
                    t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }
                else if (geom.type == TRIANGLE)
                {
                    t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }

                // Compute the minimum t from the intersection tests to determine what
                // scene geometry object was hit first.
                if (t > 0.0f && t_min > t)
                {
                    t_min = t;
                    hit_geom_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                }
            }
            continue;
        }
      
        // Check intersection with left and right children
        bool hitL = bboxIntersectionTest(nodes[node.l].bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times);
        bool hitR = bboxIntersectionTest(nodes[node.r].bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times);

        if (hitL && hitR) {
            // Both hit
            nodeStack[nodeStackFinger] = node.l;
            nodeStackFinger++;
            nodeStack[nodeStackFinger] = node.r;
            nodeStackFinger++;
        } else if (hitR) {
            nodeStack[nodeStackFinger] = node.r;
            nodeStackFinger++;
        } else if (hitL) {
            nodeStack[nodeStackFinger] = node.l;
            nodeStackFinger++;
        }
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
        intersections[path_index].materialId = -1;
    }
    else
    {
        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int numPaths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;
    if (pathSegments[idx].remainingBounces < 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) // if the intersection exists...
    {
        // Set up the RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

        // Scatter the ray at intersecting point and perform bsdf evaluation
        scatterRay(pathSegments[idx], 
            pathSegments[idx].ray.origin + 
            pathSegments[idx].ray.direction * intersection.t, 
            intersection.surfaceNormal, 
            materials[intersection.materialId],
            rng);
    }
    else {
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
        pathSegments[idx].color = glm::vec3(0.0f);
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
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
    PathSegment* dev_path_end = dev_paths + pixelcount;
    ShadeableIntersection* dev_intersections_end = dev_intersections + pixelcount;
    int remaining_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    // Terminate iteration once maximum trace depth has been reached or if no valid rays remain
    while (remaining_paths && depth < traceDepth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (remaining_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            remaining_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_nodes,
            hst_scene->nodes.size(),
            hst_scene->bvh.rootIdx,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if PATHTRACE_CONTIGUOUS_MATERIALID
        // Compaction #1 : Terminate paths with invalid intersections
        // Order arrays in decreasing materialId order
        thrust::stable_sort_by_key(
            thrust::device, 
            dev_intersections, 
            dev_intersections_end, 
            dev_paths,
            [] __device__(const ShadeableIntersection & si1, const ShadeableIntersection & si2) { return si1.materialId > si2.materialId; });

        // Now that invalid intersections with materialId == -1 is at the end of the array,
        // find the start index of invalid intersections using partition_point
        dev_intersections_end = thrust::partition_point(
            thrust::device,
            dev_intersections,
            dev_intersections_end,
            [] __device__(const ShadeableIntersection & si) { return si.materialId > -1; });

        // If no ray remain after intersection tests, break
        if (!(dev_intersections_end - dev_intersections)) break;
#endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            remaining_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        checkCUDAError("shade material error");

        // Compaction #2 : Terminate paths with no more remaining bounces
        dev_path_end = thrust::stable_partition(
            thrust::device, 
            dev_paths, 
            dev_path_end, 
            [] __device__ (const PathSegment& ps) { return ps.remainingBounces > -1; });

        remaining_paths = dev_path_end - dev_paths;

        if (guiData)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
