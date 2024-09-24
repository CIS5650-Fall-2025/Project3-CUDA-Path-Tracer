#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

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
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhnodes = NULL;
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

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (scene->meshes.size() > 0) {
        int num_tris = scene->triangle_count;
        cudaMalloc(&dev_triangles, sizeof(Triangle) * num_tris);
        cudaMemcpy(dev_triangles, scene->mesh_triangles.data(), num_tris * sizeof(Triangle), cudaMemcpyHostToDevice);

        int num_bvhnodes = scene->bvhNodes.size();
        cudaMalloc(&dev_bvhnodes, sizeof(BVHNode) * num_bvhnodes);
        cudaMemcpy(dev_bvhnodes, scene->bvhNodes.data(), num_bvhnodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_triangles);
    cudaFree(dev_bvhnodes);

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
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
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
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    Triangle* tris,
    int num_tris,
    BVHNode* bvhnodes)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE) {
                //t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH) {
#define USE_BVH 1
#if USE_BVH
                t = bvhIntersectionTest(pathSegment.ray, tmp_intersect, tmp_normal, outside, bvhnodes, tris, num_tris);
#else
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tris, num_tris);
#endif
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

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__global__ void shadeMaterials(int iter,
                               int num_paths,
                               int depth,
                               ShadeableIntersection* shadeableIntersections,
                               PathSegment* pathSegments,
                               Material* materials) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > num_paths || pathSegments[idx].remainingBounces <= 0) {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) // if the intersection exists...
    {
        // Set up the RNG
        // LOOK: this is how you use thrust's RNG! Please look at
        // makeSeededRandomEngine as well.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;
        PathSegment& curr_seg = pathSegments[idx];
        Ray& curr_ray = curr_seg.ray;

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0f) {
            curr_seg.color *= (materialColor * material.emittance);
            curr_seg.remainingBounces = 0;
        } 
        else  if (material.specular.isSpecular == false) {
            //perfectly diffuse for now
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            glm::vec3 wi;
            scatterRay(curr_seg, isect_pt, intersection.surfaceNormal, material, rng, wi);

            wi = glm::normalize(wi);

            float theta = glm::acos(glm::dot(wi, intersection.surfaceNormal));
            float costheta = glm::cos(theta);
            float pdf = costheta * INV_PI;
            if (pdf == 0.f) {
                curr_seg.remainingBounces = 0;
                return;
            }

            glm::vec3 bsdf = materialColor * INV_PI;
            float lambert = glm::abs(glm::dot(wi, intersection.surfaceNormal));

            curr_seg.color *= (bsdf * lambert) / pdf;

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
        }
        else {
            //perfectly specular for now
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            glm::vec3 wi = glm::reflect(curr_ray.direction, intersection.surfaceNormal);

            wi = glm::normalize(wi);

            glm::vec3 bsdf = materialColor * INV_PI;
            float lambert = glm::abs(glm::dot(wi, intersection.surfaceNormal));

            curr_seg.color *= (bsdf * lambert); //pdf = 1

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
        }
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
    }
    else {
        pathSegments[idx].color = glm::vec3(0.0f);
        pathSegments[idx].remainingBounces = 0;
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

struct ShouldTerminate {
    __host__ __device__ bool operator()(const PathSegment& x)
    {
        return x.remainingBounces > 0;
    }
};

struct CompareMaterials
{
    __host__ __device__ bool operator()(const ShadeableIntersection& first, const ShadeableIntersection& second)
    {
        return first.materialId < second.materialId;
    }
};

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
    int num_paths = dev_path_end - dev_paths; //just the pixel count for now

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_triangles,
            hst_scene->triangle_count,
            dev_bvhnodes
        );
        checkCUDAError("compute intersections");
        cudaDeviceSynchronize();
        depth++;

#define SORTBYMATERIAL 1
#if SORTBYMATERIAL
        thrust::device_ptr<ShadeableIntersection> dev_inters_to_sort(dev_intersections);
        thrust::device_ptr<PathSegment> dev_paths_to_sort(dev_paths); //values
        thrust::stable_sort_by_key(dev_inters_to_sort, dev_inters_to_sort + num_paths, dev_paths_to_sort, CompareMaterials());
#endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials
            );

#define USE_COMPACTION 1
#if USE_COMPACTION
        thrust::device_ptr<PathSegment> dev_paths_to_compact(dev_paths);
        thrust::device_ptr<PathSegment> last_elt = thrust::stable_partition(thrust::device, dev_paths_to_compact, dev_paths_to_compact + num_paths, ShouldTerminate());
        num_paths = last_elt.get() - dev_paths;
#endif

        iterationComplete = (depth >= traceDepth || num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //NOTE: changed N to pixelcount from num paths, still want to check paths that will be terminated
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
