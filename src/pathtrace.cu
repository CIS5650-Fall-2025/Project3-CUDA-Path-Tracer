#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "oidn.hpp"

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
static glm::vec3* dev_normals = NULL;
static glm::vec3* dev_albedos = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
#ifdef NDEBUG
static oidn::DeviceRef oidnDevice = NULL;
#endif

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

	cudaMalloc(&dev_normals, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_normals, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_albedos, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_albedos, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need


    checkCUDAError("pathtraceInit");

    // Initialize OIDN
#ifdef NDEBUG
	oidnDevice = oidn::newDevice(oidn::DeviceType::CUDA);
	oidnDevice.commit();
#endif
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_normals);
	cudaFree(dev_albedos);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* TODO: motion blur - jitter rays "in time"
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Antialiasing jitter - randomizes the ray direction within a pixel
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
		thrust::uniform_real_distribution<float> unifDist(-0.5, 0.5);
		float jitterX = unifDist(rng);
		float jitterY = unifDist(rng);

        glm::vec3 rayDirection  = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

		// Depth of field jitter - randomizes the ray origin within the aperture
		// First, calculate point on the lens as a random point within the aperture
        thrust::uniform_real_distribution<float> apertureDist(-1.0, 1.0);
		glm::vec3 lensPoint = cam.position + (cam.apertureRadius * 2.0f * (cam.up * apertureDist(rng) + cam.right * apertureDist(rng)));

		// Next, calculate point of focus on the focal plane
		glm::vec3 focusPoint = cam.position + (cam.focalLength * rayDirection);

		// Finally, calculate the ray direction from the lens point to the focus point
		segment.ray.direction = glm::normalize(focusPoint - lensPoint);
		segment.ray.origin = lensPoint;

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
    Material* materials,
    glm::vec3* normals,
    glm::vec3* albedos)
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

			if (depth == 0) {
				normals[path_index] += normal;
				albedos[path_index] += materials[geoms[hit_geom_index].materialid].color;
			}
        }
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
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t <= 0.0f) {
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = 0;
		return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Material material = materials[intersection.materialId];
    glm::vec3 materialColor = material.color;

    // If the material indicates that the object was a light, "light" the ray
    if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
		pathSegments[idx].remainingBounces = 0;
        return;
    }

	glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
	scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, const PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= nPaths) return;
 
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
}

struct MaterialIdComparator
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const
    {
        return a.materialId < b.materialId;
    }
};

#ifdef NDEBUG
void applyOIDNDenoise(glm::vec3* output, glm::vec3* colors, glm::vec3* normals, glm::vec3* albedos, glm::ivec2 resolution)
{
    // Create OIDN filter
    oidn::FilterRef filter = oidnDevice.newFilter("RT"); // Generic ray tracing filter

	// Prefilter normals and albedos
    filter.setImage("color", normals, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("output", normals, oidn::Format::Float3, resolution.x, resolution.y);
    filter.commit();
    filter.execute();

    filter.set("hdr", true); // Assuming the image is in HDR 
    filter.setImage("color", albedos, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("output", albedos, oidn::Format::Float3, resolution.x, resolution.y);
    filter.commit();
    filter.execute();

    filter.setImage("color", colors, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("output", output, oidn::Format::Float3, resolution.x, resolution.y);
	filter.setImage("normal", normals, oidn::Format::Float3, resolution.x, resolution.y);
	filter.setImage("albedo", albedos, oidn::Format::Float3, resolution.x, resolution.y);
    filter.commit();
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (oidnDevice.getError(errorMessage) != oidn::Error::None)
    {
        fprintf(stderr, "Error: %s\n", errorMessage);
    }
}
#endif

__global__ void renormalizeNormals(int n, glm::vec3* normals)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= n) return;

	normals[index] = glm::normalize(normals[index]);
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, int maxIterations)
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    int depth = 0;
	while (num_paths > 0 && depth < traceDepth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_materials,
            dev_normals,
            dev_albedos
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // Before shading, sort by material to increase memory coherency and reduce warp divergence.
        thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths,
			MaterialIdComparator()
        );

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

		// Compact streams with thrust partition (reorders paths so that terminated paths are at the end)
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end,
            [] __device__(const PathSegment& segment) {
                return segment.remainingBounces != 0;
            }
        );

		checkCUDAError("remove terminated paths");
        cudaDeviceSynchronize();

		num_paths = dev_path_end - dev_paths;
        ++depth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

#ifdef NDEBUG
    // Apply OIDN denoising
    if (iter == maxIterations) {
        renormalizeNormals<<<numBlocksPixels, blockSize1d>>> (pixelcount, dev_normals);
        applyOIDNDenoise(dev_image, dev_image, dev_normals, dev_albedos, cam.resolution);
	}
#endif

	// Send image to OpenGL for display
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
