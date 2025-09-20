#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "bsdf.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "ImGui/imgui.h"

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

__global__ void accumulateAlbedoNormal(int num_paths, ShadeableIntersection* intersections, Material* materials, 
    glm::vec3* albedo_image, glm::vec3* normal_image)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_paths)
    {
        auto& inter = intersections[index];
        if (inter.t > 0.0f)
        {
            auto& mat = materials[inter.materialId];

			albedo_image[index] += glm::vec3(mat.albedo);
            normal_image[index] += inter.surfaceNormal;
        }
    }
}

__global__ void normalizeAlbedoNormal(glm::ivec2 resolution, int iter, glm::vec3* albedo_image, glm::vec3* normal_image,
    glm::vec3* out_albedo, glm::vec3* out_normal)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + y * resolution.x;

        out_albedo[index] = albedo_image[index] / static_cast<float>(iter);
        out_normal[index] = normal_image[index] / static_cast<float>(iter);
    }
}

static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* dev_image = nullptr;

// Could be HDR so use float3
static glm::vec3* in_dev_image_denoise = nullptr;
static glm::vec3* out_dev_image_denoise = nullptr;

static glm::vec3* accumulate_albedo = nullptr;
static glm::vec3* accumulate_normal = nullptr;
static glm::vec3* dev_image_albedo = nullptr;
static glm::vec3* dev_image_normal = nullptr;

static Geom* dev_geoms = nullptr;
static Material* dev_materials = nullptr;

static PathSegment* dev_paths = nullptr;

static ShadeableIntersection* dev_intersections = nullptr;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static OptixDeviceContext ctx{};
static OptixDenoiser denoiser{};
static OptixDenoiserSizes denoiser_sizes{};
static CUdeviceptr d_state = NULL, d_scratch = NULL;

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

	cudaMalloc(&in_dev_image_denoise, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&out_dev_image_denoise, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&accumulate_albedo, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&accumulate_normal, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_image_albedo, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_image_normal, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");

    optixInit();
    OptixResult result = optixDeviceContextCreate(nullptr, nullptr, &ctx);
    if (result != OPTIX_SUCCESS) 
    {
        printf("OptiX device context creation failed: %d\n", result);
    }

    OptixDenoiserOptions o{};
	o.guideAlbedo = 1;
	o.guideNormal = 1;
    // TODO: alpha denoise?
    result = optixDenoiserCreate(ctx, OPTIX_DENOISER_MODEL_KIND_HDR, &o, &denoiser);
    if (result != OPTIX_SUCCESS) 
    {
        printf("OptiX denoiser creation failed: %d\n", result);
    }

    optixDenoiserComputeMemoryResources(denoiser, cam.resolution.x, cam.resolution.y, &denoiser_sizes);

    void* state;
	void* scratch;

    cudaMalloc(&state, denoiser_sizes.stateSizeInBytes);
    cudaMalloc(&scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes);

    d_state = reinterpret_cast<CUdeviceptr>(state);
    d_scratch = reinterpret_cast<CUdeviceptr>(scratch);

    result = optixDenoiserSetup(denoiser, nullptr, cam.resolution.x, cam.resolution.y, 
        d_state, denoiser_sizes.stateSizeInBytes, d_scratch, 
        denoiser_sizes.withoutOverlapScratchSizeInBytes);
    if (result != OPTIX_SUCCESS) 
    {
        printf("OptiX denoiser setup failed: %d\n", result);
    }
}

void pathtraceReset(const Scene& scene)
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(accumulate_albedo, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(accumulate_normal, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_albedo, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_normal, 0, pixelcount * sizeof(glm::vec3));
	checkCUDAError("pathTraceReset");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(in_dev_image_denoise);
    cudaFree(out_dev_image_denoise);

    cudaFree(accumulate_albedo);
    cudaFree(accumulate_normal);
    cudaFree(dev_image_albedo);
    cudaFree(dev_image_normal);

    checkCUDAError("pathtraceFree");

    if (ctx)
    {
        optixDeviceContextDestroy(ctx);
        cudaFree(reinterpret_cast<void*>(d_state));
        cudaFree(reinterpret_cast<void*>(d_scratch));
    }
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

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, 0, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
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
    ShadeableIntersection* intersections)
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
            // TODO: add more intersection tests here... triangle? metaball? CSG?

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

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, const PathSegment* iterationPaths)

{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// Pass by value or else memory doesn't read right
__global__ void average(glm::vec3* in_image, glm::ivec2 resolution, int iter, glm::vec3* out_image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        out_image[index] = in_image[index] / static_cast<float>(iter);
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, const int frame, int iter, const bool sort, const DisplayMode displayMode, const bool saveImage)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(16, 16);
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

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    uint32_t num_paths = dev_path_end - dev_paths;
    const auto initial_num_paths = num_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

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
            dev_intersections
        );

        if (depth == 0)
        {
            accumulateAlbedoNormal<<<numBlocksPixels, blockSize1d>>>(
	            pixelcount, dev_intersections, dev_materials, accumulate_albedo, accumulate_normal);
        }

        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        if (sort)
        {
            thrust::sort_by_key(thrust::device,
                dev_intersections,
                dev_intersections + num_paths,
                dev_paths,
                [] __device__(const ShadeableIntersection & a, const ShadeableIntersection & b) {
                return a.materialId < b.materialId;
            });
        }

        shade<<<numblocksPathSegmentTracing, blockSize1d>>>(
	        depth,
	        iter,
	        num_paths,
	        dev_intersections, dev_materials, dev_paths);

        auto newEnd = thrust::partition(thrust::device, dev_paths,dev_paths + num_paths,
            [] __device__(const PathSegment& p)
            {
                return p.remainingBounces > 0;
            });
        num_paths = static_cast<uint32_t>(newEnd - dev_paths);

        iterationComplete = num_paths == 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    finalGather<<<numBlocksPixels, blockSize1d>>>(initial_num_paths, dev_image, dev_paths);

    normalizeAlbedoNormal<<<blocksPerGrid2d, blockSize2d>>>(
        cam.resolution, iter, accumulate_albedo, accumulate_normal, dev_image_albedo, dev_image_normal);

    ///////////////////////////////////////////////////////////////////////////

    const auto frameToDenoise = iter % frame == 0;
    // Denoise every N frames
    if (frameToDenoise || saveImage)
    {
		// Divide overall image by number of iterations
		average<<<blocksPerGrid2d, blockSize2d>>>(dev_image, cam.resolution, iter, in_dev_image_denoise);

        OptixImage2D in{};
		in.data = reinterpret_cast<CUdeviceptr>(in_dev_image_denoise);
		in.width = cam.resolution.x;
		in.height = cam.resolution.y;
		in.rowStrideInBytes = sizeof(glm::vec3) * cam.resolution.x;
		in.pixelStrideInBytes = sizeof(glm::vec3);
		in.format = OPTIX_PIXEL_FORMAT_FLOAT3;

        OptixImage2D out = in;
		out.data = reinterpret_cast<CUdeviceptr>(out_dev_image_denoise);

        OptixImage2D albedo{};
		albedo.data = reinterpret_cast<CUdeviceptr>(dev_image_albedo);
		albedo.width = cam.resolution.x;
		albedo.height = cam.resolution.y;
		albedo.rowStrideInBytes = sizeof(glm::vec3) * cam.resolution.x;
		albedo.pixelStrideInBytes = sizeof(glm::vec3);
		albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;

        OptixImage2D normal = albedo;
		normal.data = reinterpret_cast<CUdeviceptr>(dev_image_normal);

        OptixDenoiserGuideLayer gl{};
		gl.albedo = albedo;
		gl.normal = normal;

        OptixDenoiserLayer ly{};
    	ly.input = in;
    	ly.output = out;

        OptixDenoiserParams p{};

        auto denoise_result = optixDenoiserInvoke(denoiser, nullptr, &p, d_state, denoiser_sizes.stateSizeInBytes, 
            &gl, &ly, 1, 0, 0, d_scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes);
        
        if (denoise_result != OPTIX_SUCCESS) 
        {
            printf("OptiX denoiser invoke failed: %d\n", denoise_result);
        }
    }

    switch (displayMode)
    {
    case PROGRESSIVE:
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
        break;
    case ALBEDO:
		sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, 1, dev_image_albedo);
        break;
	case NORMAL:
		sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, 1, dev_image_normal);
        break;
    case DENOISED:
		sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, 1, out_dev_image_denoise);
        break;
    }

    if (saveImage)
    {
        cudaMemcpy(hst_scene->state.image.data(), out_dev_image_denoise,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }
}
