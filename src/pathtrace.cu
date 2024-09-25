#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static GuiDataContainer* guiData = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static glm::vec3* dev_vertices = NULL;
static Mesh* dev_meshes = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec2* dev_texcoords = NULL;

#if BVH
static BVHNode* dev_bvh = NULL;
#endif

#if OIDN
#define EMA_ALPHA 0.2f
#define DENOISE_INTERVAL 100

#include <OpenImageDenoise/oidn.hpp>

static glm::vec3* dev_denoised = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;

void denoise()
{
    int width = hst_scene->state.camera.resolution.x,
        height = hst_scene->state.camera.resolution.y;

    // Create an Intel Open Image Denoise device
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create a filter for denoising a beauty (color) image using prefiltered auxiliary images too
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color", dev_image, oidn::Format::Float3, width, height); // beauty
    filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height); // auxiliary
    filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height); // auxiliary
    filter.setImage("output", dev_denoised, oidn::Format::Float3, width, height); // denoised beauty
    filter.set("hdr", true); // image is HDR
    filter.set("cleanAux", true); // auxiliary images will be prefiltered
    filter.commit();

    // Create a separate filter for denoising an auxiliary albedo image (in-place)
    oidn::FilterRef albedoFilter = device.newFilter("RT"); // same filter type as for beauty
    albedoFilter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.setImage("output", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.commit();

    // Create a separate filter for denoising an auxiliary normal image (in-place)
    oidn::FilterRef normalFilter = device.newFilter("RT"); // same filter type as for beauty
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
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;
}

__global__
void copyFirstTraceResult(
    PathSegment* pathSegments, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    glm::vec3* albedo, glm::vec3* normal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        PathSegment pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];

        albedo[pathSegment.pixelIndex] = pathSegment.color;
        normal[pathSegment.pixelIndex] = intersection.surfaceNormal;
    }
}

__global__
void emaMergeDenoisedAndImage(int pixelcount, glm::vec3* image, glm::vec3* denoised)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelcount) {
        // exponential moving average
        image[idx] = image[idx] * (1 - EMA_ALPHA) + denoised[idx] * EMA_ALPHA;
    }
}
#endif

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

struct isRayAlive
{
    __host__ __device__ bool operator()(const PathSegment& path)
    {
        return path.remainingBounces > 0;
    }
};

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

    //Mesh data
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Mesh));
    cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_texcoords, scene->texcoords.size() * sizeof(glm::vec2));
    cudaMemcpy(dev_texcoords, scene->texcoords.data(), scene->texcoords.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

#if BVH
    cudaMalloc(&dev_bvh, scene->bvh.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvh, scene->bvh.data(), scene->bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
#endif

#if OIDN
    cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image); 
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_texcoords);
    cudaFree(dev_normals);
    cudaFree(dev_vertices);
    cudaFree(dev_meshes);

#if BVH
    cudaFree(dev_bvh);
#endif

#if OIDN
    cudaFree(dev_denoised);
    cudaFree(dev_albedo);
    cudaFree(dev_normal);
#endif


    checkCUDAError("pathtraceFree");
}

__device__ glm::vec3 checkerboard(glm::vec2 uv)
{
    if ((int)(uv.x * 10) % 2 == (int)(uv.y * 10) % 2)
        return glm::vec3(.2f);
    else
        return glm::vec3(.8f);
}
__device__ glm::vec3 palettes(glm::vec2 uv)
{
    glm::vec3 a(0.5, 0.5, 0.5), b(0.5, 0.5, 0.5), c(1.0, 1.0, 1.0), d(0.00, 0.33, 0.67);
    return a + b * glm::cos(TWO_PI * (c * glm::length(uv) + d));
}

__host__ __device__
glm::vec2 ConcentricSampleDisk(const glm::vec2& u)
{
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f, 1.0f);

    if (uOffset.x == 0.0f && uOffset.y == 0.0f)
    {
        return glm::vec2(0.0f, 0.0f);
    }

    float theta, r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y))
    {
        r = uOffset.x;
        theta = PI_OVER_FOUR * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = PI_OVER_TWO - PI_OVER_FOUR * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(glm::cos(theta), glm::sin(theta));
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

        //stochastic sampled antialiasing (SSAA)
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );

        // Depth of field automatically Enabled for camera with LENSRADIUS and FOCALDIS
        if (cam.lensRadius > 0)
        {
            glm::vec2 pLens = cam.lensRadius * ConcentricSampleDisk(glm::vec2(u01(rng), u01(rng)));
            float ft = cam.focalDistance / glm::dot(cam.view, segment.ray.direction);
            glm::vec3 pFocus = segment.ray.origin + segment.ray.direction * ft;
            segment.ray.origin += cam.right * pLens.x + cam.up * pLens.y;
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
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
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
#if BVH
    BVHNode* bvh,
#endif
    Mesh* meshes
    , glm::vec3* vertices
    , glm::vec3* normals
    , glm::vec2* texcoords)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

        int tmp_material_index;
        glm::vec2 tmp_texcoord;

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside,tmp_uv);
            }
            else if (geom.type == MESH) 
            {
#if BVH
                t = meshIntersectionTestBVH(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv,
                    bvh, meshes, vertices, normals, texcoords, tmp_material_index);

#else
                t = meshIntersectionTestNaive(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv, 
                    meshes, vertices, normals, texcoords, tmp_material_index);
#endif        
            }
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
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
            intersections[path_index].uv = uv;
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
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                switch (material.procedualTextureID) {
                     case 1: material.color = checkerboard(intersection.uv); break;
                     case 2: material.color = palettes(intersection.uv); break;
                     default: break;
                }
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                glm::vec3 intersect = intersection.t * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
            }
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
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
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

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
#if BVH
            , dev_bvh
#endif 
            , dev_meshes, dev_vertices, dev_normals, dev_texcoords
        );
        cudaDeviceSynchronize();
        depth++;

        // --- Shading Stage ---
#ifndef SORT_MATERIAL_ID
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialsCmp());
#endif
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

#if OIDN
        if (depth == 1 && (iter % DENOISE_INTERVAL == 0 || iter == hst_scene->state.iterations))
            copyFirstTraceResult << <numblocksPathSegmentTracing, blockSize1d >> > (
                dev_paths, num_paths, dev_intersections, dev_albedo, dev_normal);
#endif

#ifdef STREAM_COMPACTION
        num_paths = thrust::partition(thrust::device,
            dev_paths, dev_paths + num_paths, isRayAlive()) - dev_paths;
#endif

        iterationComplete = depth == traceDepth || num_paths == 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

#ifdef STREAM_COMPACTION
    num_paths = dev_path_end - dev_paths;
#endif

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

#if OIDN
    if (iter % DENOISE_INTERVAL == 0 && iter != 0)
    {
        denoise();
        emaMergeDenoisedAndImage << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_denoised);
    }
    else if (iter == hst_scene->state.iterations)
    {
        denoise();
        std::swap(dev_image, dev_denoised);
    }
#endif

    // --- Rendering Stage ---

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
