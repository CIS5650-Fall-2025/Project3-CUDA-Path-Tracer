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
#include "OpenImageDenoise/oidn.hpp"

#include <device_launch_parameters.h>


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

// ---------------------------------------------------------
// Static Variables for Device Memory

// host variables 
static Scene* hst_scene = NULL;
static std::vector<cudaTextureObject_t> hst_texObjs;

static GuiDataContainer* guiData = NULL;

// device variables
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;

static cudaTextureObject_t* dev_texObjs;
static std::vector<cudaArray_t> dev_texData;

#if BVH 
static BVHNode* dev_BVHNodes = NULL;
static int* dev_BVHTriIdx = NULL;
#endif

#if DENOISE 
static glm::vec3* dev_denoised_image = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
#endif

// -----------------------------------------------------------

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// Most of this code was taken from NVIDIA programming guide: (3.2.14.1.1. Texture Object API)
// link: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
void createTexObjs(int i) {

    // Specify texture
    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_texData[i];

    // Specify texture object parameters
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Allocate result of transformation in device memory (in this case copying to dev)
    cudaMemcpy(&dev_texObjs[i], &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
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

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_texObjs, scene->textures.size() * sizeof(cudaTextureObject_t));
    dev_texData.resize(scene->textures.size());

    for (int i = 0; i < scene->textures.size(); i++) {

        // channel descriptor
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

        cudaMallocArray(&dev_texData[i], &channelDesc, scene->textures[i].width, scene->textures[i].height);
        cudaMemcpyToArray(dev_texData[i],
            0,
            0,
            scene->textures[i].data,
            scene->textures[i].channels * scene->textures[i].width * scene->textures[i].height * sizeof(unsigned char),
            cudaMemcpyHostToDevice);

        createTexObjs(i);
    }

#if BVH 
    cudaMalloc(&dev_BVHNodes, hst_scene->bvhNode.size() * sizeof(BVHNode));
    cudaMemcpy(dev_BVHNodes, scene->bvhNode.data(), hst_scene->bvhNode.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_BVHTriIdx, hst_scene->triangles.size() * sizeof(int));
    cudaMemcpy(dev_BVHTriIdx, hst_scene->triIdx.data(), hst_scene->triangles.size() * sizeof(int), cudaMemcpyHostToDevice);
#endif

#if DENOISE 
    // denoise buffer data
    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
#endif

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

    for (int i = 0; i < hst_texObjs.size(); i++) {
        cudaDestroyTextureObject(hst_texObjs[i]);
        cudaFreeArray(dev_texData[i]);
    }

#if BVH
    cudaFree(dev_BVHNodes);
    cudaFree(dev_BVHTriIdx);
#endif

#if DENOISE 
    cudaFree(dev_denoised_image);
    cudaFree(dev_albedo);
    cudaFree(dev_normal);
#endif

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

        // implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        thrust::uniform_real_distribution<float> u02(0, 1);

        // toggle antialiasing here
#if AA
        x += u01(rng);
        y += u02(rng);
#endif

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)

        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// computeIntersections handles generating ray intersections.
// Generating new rays is handled in your shader(s).
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    Triangle* triangles,
    cudaTextureObject_t* textureObjs,
    Texture* dev_textures
#if BVH
    ,BVHNode* bvhNodes,
    int* bvhTriIdx
#endif
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t = 0.f;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec2 uv;
        glm::vec2 tmp_uv;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        
        int geomIdx = -1;

#if !BVH
        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++)
        {
            t = 0.f;
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
            
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;

                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
            }
        }
#else 
        // bvh parse through global objs
        t = BVHIntersectionTest(
            pathSegment.ray,
            tmp_intersect,
            tmp_normal,
            tmp_uv,
            bvhNodes,
            triangles,
            bvhTriIdx,
            geomIdx,
            outside);

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = geomIdx;

            intersect_point = tmp_intersect;
            normal = tmp_normal;
            uv = tmp_uv;
        }
#endif
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
            intersections[path_index].textureId = geoms[hit_geom_index].texIdx;
            intersections[path_index].uv = uv;
        }
    }
}

// Shading kernel with BSDF evaluation 
__global__ void shadeBSDFMaterial(
    int depth,
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* textureObjs
#if DENOISE 
    ,glm::vec3* dev_albedo,
    glm::vec3* dev_normal
#endif
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0.f;
            }

            // BSDF evaluation
            else {

                // find point at which ray hits surface 
                glm::vec3 intersect = pathSegments[idx].ray.direction * intersection.t + pathSegments[idx].ray.origin;

                scatterRay(pathSegments[idx],
                    intersect,
                    intersection.surfaceNormal,
                    material,
                    rng);

                cudaTextureObject_t texObj = NULL;
                float4 uvColor;
                glm::vec3 texCol;

                // calculate and assign texture color according to if mesh has texture
                if (intersection.textureId != -1) {

                    if (intersection.textureId == -2 || intersection.textureId == -3) {
                        texCol = generateProceduralTexture(intersection);
                    }
                    else {
                        texObj = textureObjs[intersection.textureId];
                        uvColor = tex2D<float4>(texObj, intersection.uv[0], intersection.uv[1]);
                        texCol = glm::vec3(uvColor.x, uvColor.y, uvColor.z);
                    }
                }

                pathSegments[idx].color *= (intersection.textureId == -1) ? material.color : texCol;
                pathSegments[idx].remainingBounces--;

#if DENOISE 
                // calculate the albedo and normal data and store them directly
                dev_albedo[pathSegments[idx].pixelIndex] = pathSegments[idx].color;
                dev_normal[pathSegments[idx].pixelIndex] = intersection.surfaceNormal;  
#endif
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0.f;
            return;
        }
    }
}

#if DENOISE 
__global__
void blendDenoised(glm::vec3* image, glm::vec3* denoised, int pixelCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelCount) {
        image[idx] = image[idx] * (1 - 0.5f) + denoised[idx] * 0.5f;
    }
}

// Basic denoising functon from OIDN documentation
// source: https://www.openimagedenoise.org/documentation.html
void applyDenoising() {
    const int width = hst_scene->state.camera.resolution.x;
    const int height = hst_scene->state.camera.resolution.y;

    // Create an Open Image Denoise device
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create a filter for denoising a beauty (color) image using optional auxiliary images too
    oidn::FilterRef filter = device.newFilter("RT");  // generic ray tracing filter
    filter.setImage("color", dev_image, oidn::Format::Float3, width, height); // beauty
    filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    filter.setImage("output", dev_denoised_image, oidn::Format::Float3, width, height);

    filter.set("hdr", true); // beauty image is HDR
    filter.set("cleanAux", true); // auxiliary images will be prefiltered
    filter.commit();

    // Create separate filters for denoising auxiliary albedo & normal image (in-place)

    oidn::FilterRef albedoFilter = device.newFilter("RT"); // same filter type as for beauty
    albedoFilter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.setImage("output", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.commit();

    oidn::FilterRef normalFilter = device.newFilter("RT"); 
    normalFilter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.setImage("output", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.commit();

    // Prefilter the auxiliary images

    albedoFilter.execute();
    normalFilter.execute();

    // Filter the beauty image
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None) {
        std::cerr << "Error! " << errorMessage << std::endl;
    }
}
#endif

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

struct hasBounces {
    __host__ __device__ bool operator()(const PathSegment& path)
    {
        return path.remainingBounces > 0;
    }
};

struct sortIntersectsByMaterial
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& p1, const ShadeableIntersection& p2)
    {
        return p1.materialId < p2.materialId;
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
    //   * DONE: Stream compact away all of the terminated paths.
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

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int total_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_triangles,
            dev_texObjs,
            dev_textures
#if BVH
            ,dev_BVHNodes,
            dev_BVHTriIdx
#endif
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        // Toggle material sort
#if MATSORT
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortIntersectsByMaterial());
#endif

        shadeBSDFMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_texObjs
#if DENOISE 
            ,dev_albedo,
            dev_normal
#endif
            );
        checkCUDAError("shade BSDF");
        cudaDeviceSynchronize();

        // based off stream compaction results.
        PathSegment* new_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, hasBounces());
        num_paths = new_path_end - dev_paths;

        if (num_paths <= 0) iterationComplete = true;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (total_paths, dev_image, dev_paths);

#if DENOISE

    // Apply denoising
    if (iter == hst_scene->state.iterations || iter % DENOISE_INTERVAL == 0) {
        applyDenoising();
        blendDenoised << <numBlocksPixels, blockSize1d >> > (dev_image, dev_denoised_image, pixelcount);
    }
    checkCUDAError("denoising");

#endif

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}