#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "mathUtils.h"

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
        pix /= iter;
        pix = math::ACESMapping(pix);
        pix = math::gammaCorrect(pix);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__device__ void spawnRay(Ray& ray, const glm::vec3& ori, const glm::vec3& dir)
{
    ray.origin = ori + EPSILON * dir;
    ray.direction = dir;
}

__global__ void getFocalDistance(SceneDev* scene, Camera* cam, float xPos, float yPos)
{
    Ray r;
    ShadeableIntersection isect;
    xPos -= (float)cam->resolution.x * 0.5f;
    yPos -= (float)cam->resolution.y * 0.5f;
    cam->generateRay(r, xPos, yPos);
    scene->intersect(r, isect);
    if (isect.t > 0.f) cam->focalLength = isect.t;
}

__global__ void generateGbuffer(SceneDev* scene, Material* materials, Camera cam, glm::vec3* albedo, glm::vec3* normal)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {

        // gen rays
        int index = x + (y * cam.resolution.x);
        PathSegment segment;

        thrust::default_random_engine rng = makeSeededRandomEngine(0, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.radiance = glm::vec3(0.f);

        glm::vec2 offset = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
        float xPix = (float)x - (float)cam.resolution.x * 0.5f + offset.x;
        float yPix = (float)y - (float)cam.resolution.y * 0.5f + offset.y;
        //cam.generateRayLens(segment.ray, xPix, yPix, u01(rng), u01(rng));
        cam.generateRay(segment.ray, xPix, yPix);

        segment.pixelIndex = index;
        
        // do intersection
        ShadeableIntersection isect;
        scene->intersect(segment.ray, isect);

        // if no hit event, sample env map
        if (isect.t < 0.f)
        {
            albedo[index] = scene->getEnvColor(segment.ray.direction);
            normal[index] = glm::vec3(0);
        }
        else
        {
            Material material = materials[isect.materialId];
            material.createMaterialInst(material, isect.uv);
            albedo[index] = material.albedo;
            normal[index] = isect.nor;
        }

    }
}

struct CompactPaths
{
    __host__ __device__ bool operator() (const PathSegment& segment)
    {
        return segment.remainingBounces == 0;
    }
};

struct CompactIsects
{
    __host__ __device__ bool operator() (const ShadeableIntersection& isect)
    {
        return isect.t < 0.f;
    }
};

struct CopyFinishedPaths
{
    __host__ __device__ bool operator() (const PathSegment& segment)
    {
        return segment.remainingBounces != 0;
    }
};

struct SortPathByKey
{
    __host__ __device__ bool operator() (const ShadeableIntersection& isect1, const ShadeableIntersection& isect2)
    {
        return isect1.materialId < isect2.materialId;
    }
};

static Scene* hst_scene = NULL;
static SceneDev* sceneDev = NULL;
static SceneDev* dev_sceneDev = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_albedoBuffer = NULL;
static glm::vec3* dev_normalBuffer = NULL;
static Camera* dev_cam = NULL;

static PathSegment* dev_paths = NULL;
static thrust::device_ptr<PathSegment> dev_paths_thrust;

static PathSegment* dev_paths_finish = NULL;
static thrust::device_ptr<PathSegment> dev_paths_finish_thrust;

static ShadeableIntersection* dev_intersections = NULL;
static thrust::device_ptr<ShadeableIntersection> dev_intersections_thrust;


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

    cudaMalloc(&dev_albedoBuffer, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedoBuffer, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normalBuffer, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normalBuffer, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    dev_paths_thrust = thrust::device_ptr<PathSegment>(dev_paths);

    cudaMalloc(&dev_paths_finish, pixelcount * sizeof(PathSegment));
    dev_paths_finish_thrust = thrust::device_ptr<PathSegment>(dev_paths_finish);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    dev_intersections_thrust = thrust::device_ptr<ShadeableIntersection>(dev_intersections);

    sceneDev = scene->sceneDev;
    cudaMalloc(&dev_sceneDev, sizeof(SceneDev));
    cudaMemcpy(dev_sceneDev, sceneDev, sizeof(SceneDev), cudaMemcpyHostToDevice);

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    generateGbuffer << <blocksPerGrid2d, blockSize2d >> > (dev_sceneDev, sceneDev->materials, cam, dev_albedoBuffer, dev_normalBuffer);

    cudaMemcpy(hst_scene->state.albedo.data(), dev_albedoBuffer,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(hst_scene->state.normal.data(), dev_normalBuffer,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);


    // auto focus
    cudaMalloc(&dev_cam, sizeof(Camera));
    cudaMemcpy(dev_cam, &scene->state.camera, sizeof(Camera), cudaMemcpyHostToDevice);
    getFocalDistance << <1, 1>> > (dev_sceneDev, dev_cam, scene->mouseClickPos.x, scene->mouseClickPos.y);
    cudaMemcpy(&scene->state.camera, dev_cam, sizeof(Camera), cudaMemcpyDeviceToHost);
    cudaFree(dev_cam);
    std::printf("new focal distance: %f at: %f, %f\n", scene->state.camera.focalLength, scene->mouseClickPos.x, scene->mouseClickPos.y);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_albedoBuffer);
    cudaFree(dev_normalBuffer);
    cudaFree(dev_paths);
    cudaFree(dev_paths_finish);
    cudaFree(dev_intersections);
    cudaFree(dev_sceneDev);
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
        PathSegment& segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.radiance = glm::vec3(0.f);

        glm::vec2 offset = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
        float xPix = (float)x - (float)cam.resolution.x * 0.5f + offset.x;
        float yPix = (float)y - (float)cam.resolution.y * 0.5f + offset.y;
        cam.generateRayLens(segment.ray, xPix, yPix, u01(rng), u01(rng));
        //cam.generateRay(segment.ray, (float)x - (float)cam.resolution.x * 0.5f + offset.x,
        //    (float)y - (float)cam.resolution.y * 0.5f + offset.y);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}


__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    SceneDev* scene,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment& segment = pathSegments[path_index];
        ShadeableIntersection isect;

        scene->intersect(segment.ray, isect);
        intersections[path_index] = isect;

        if (isect.t < 0.f)
        {
            glm::vec3 skyColor = scene->getEnvColor(segment.ray.direction);
            segment.radiance += segment.throughput * skyColor;
            segment.remainingBounces = 0;
            return;
        }

        //float t;
        //glm::vec3 intersect_point;
        //glm::vec3 normal;
        //float t_min = FLT_MAX;
        //uint32_t hit_geom_index = UINT32_MAX;
        //int materialID = -1;
        //bool outside = true;

        //glm::vec3 tmp_intersect;
        //glm::vec3 tmp_normal;

        //// naive parse through global geoms
        //for (uint32_t i = 0; i < scene->primNum; ++i)
        //{
        //    uint32_t primID = scene->primitives[i].primId;
        //    if (primID < scene->triNum)
        //    {
        //        glm::vec3 bary;
        //        t = triangleIntersection(pathSegment.ray,
        //            scene->vertices[3 * primID], scene->vertices[3 * primID + 1], scene->vertices[3 * primID + 2], tmp_normal, bary);
        //        if (t < 0.f) continue;

        //        if (t > 0.f && t_min > t)
        //        {
        //            t_min = t;
        //            hit_geom_index = primID;
        //            materialID = scene->primitives[i].materialId;
        //            intersect_point = pathSegment.ray.origin + t * pathSegment.ray.direction;
        //            normal = scene->normals[3 * primID] * bary.x + scene->normals[3 * primID + 1] * bary.y
        //                + scene->normals[3 * primID + 2] * bary.z;
        //        }
        //    }
        //    else
        //    {
        //        Geom& geom = scene->geoms[primID - scene->triNum];

        //        if (geom.type == CUBE)
        //        {
        //            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        //        }
        //        else if (geom.type == SPHERE)
        //        {
        //            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        //        }

        //        if (t > 0.0f && t_min > t)
        //        {
        //            t_min = t;
        //            hit_geom_index = primID;
        //            materialID = scene->primitives[i].materialId;
        //            intersect_point = tmp_intersect;
        //            normal = tmp_normal;
        //        }

        //    }
        //}

        //intersections[path_index].primId = hit_geom_index;
        //if (hit_geom_index == UINT32_MAX)
        //{
        //    intersections[path_index].t = -1.0f;
        //}
        //else
        //{
        //    // The ray hits something
        //    intersections[path_index].t = t_min;
        //    intersections[path_index].materialId = materialID;
        //    intersections[path_index].nor = normal;
        //}
    }
}

__global__ void sampleSurface(
    int iter,
    int num_paths,
    SceneDev* scene,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    ShadeableIntersection isect = shadeableIntersections[idx];
    PathSegment& segment = pathSegments[idx];
    glm::vec3 hitPoint = segment.ray.origin + isect.t * segment.ray.direction;

    // case when ray hit nothing
    /*
    if (isect.primId == UINT32_MAX)
    {
        scene->sampleEnv(segment);
        segment.remainingBounces = 0;
        return;
    }
    else
    {
        segment.radiance = glm::vec3(isect.t / 100.f);
        segment.remainingBounces = 0;
        return;
    }
    */

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Material material = materials[isect.materialId];
    material.createMaterialInst(material, isect.uv);

    // case we have a light hit
    if (material.type == Light)
    {
        segment.remainingBounces = 0;
        segment.throughput *= material.albedo * material.emittance;
        segment.radiance += segment.throughput;
    }
    else
    {
        glm::vec3 rn = glm::vec3(u01(rng), u01(rng), u01(rng));
        float absCos;
        float pdf = 1.f;

        // do a light sample
        float liPdf;
        float lightWeight = 0.f;
        glm::vec3 wi;
        glm::vec3 radiance = scene->sampleEnv(hitPoint, wi, rn, &liPdf);
        if (liPdf > EPSILON)
        {
            absCos = math::clampDot(wi, isect.nor);
            glm::vec3 f = material.getBSDF(isect.nor, segment.ray.direction, wi, &pdf);
            radiance = f * radiance * absCos / liPdf;
            lightWeight = math::powerHeuristic(liPdf, pdf);
            segment.radiance += lightWeight * (segment.throughput * radiance);
        }

        // bsdf sample
        wi = glm::vec3(0.f);
        pdf = 1.f;
        glm::vec3 bsdf = material.samplef(isect.nor, segment.ray.direction, wi, rn, &pdf);

        if (pdf < EPSILON)
        {
            segment.remainingBounces = 0;
        }
        else
        {
            absCos = (material.type == Specular) ? 1.f : math::clampDot(wi, isect.nor);
            segment.throughput *= bsdf * (absCos / pdf);
            spawnRay(segment.ray, hitPoint, wi);
            --segment.remainingBounces;
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
        image[iterationPath.pixelIndex] += iterationPath.radiance;
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    thrust::device_ptr<PathSegment> finished_tail = dev_paths_finish_thrust;

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
            dev_sceneDev,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        finished_tail = thrust::remove_copy_if(dev_paths_thrust, dev_paths_thrust + num_paths, finished_tail, CopyFinishedPaths());
        thrust::remove_if(dev_intersections_thrust, dev_intersections_thrust + num_paths, CompactIsects());
        auto arrTail = thrust::remove_if(dev_paths_thrust, dev_paths_thrust + num_paths, CompactPaths());
        num_paths = arrTail - dev_paths_thrust;
        
        if (num_paths == 0) break;
        // sort rays
        //thrust::sort_by_key(dev_intersections_thrust, dev_intersections_thrust + num_paths, dev_paths_thrust, SortPathByKey());

        numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        sampleSurface<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_sceneDev,
            dev_intersections,
            dev_paths,
            sceneDev->materials
        );

        finished_tail = thrust::remove_copy_if(dev_paths_thrust, dev_paths_thrust + num_paths, finished_tail, CopyFinishedPaths());
        arrTail = thrust::remove_if(dev_paths_thrust, dev_paths_thrust + num_paths, CompactPaths());
        num_paths = arrTail - dev_paths_thrust;

        iterationComplete = (num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    num_paths = finished_tail - dev_paths_finish_thrust;
    dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths_finish);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
