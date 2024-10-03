#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)


#define USE_RUSSIAN_ROULETTE 1

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;

static Material* dev_materials = NULL;
static Geom* dev_geoms = NULL;
static Geom* dev_lights = NULL;
static Triangle* dev_geomTriangles = NULL;
static Triangle* dev_lightTriangles = NULL;
static int* dev_totalNumberOfLights = NULL;

static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

struct sortMaterialCondition
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2)
    {
        return s1.materialId < s2.materialId;
    }
};

struct has_remaining_bounces
{
    __host__ __device__
        bool operator()(const PathSegment& path)
    {
        return path.remainingBounces > 0;
    }
};

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

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void initialiseTriangles(Triangle* dev_triangles, std::vector<Geom>& geometries, const int totalNumberOfGeom)
{   
    if (totalNumberOfGeom == 0) {
        return;
    }

    int totalNumberOfTriangles = 0;
    for (int i = 0; i < totalNumberOfGeom; i++) {
        if (geometries[i].type == MESH) {
            totalNumberOfTriangles += geometries[i].numTriangles;
        }
    }

    if (totalNumberOfTriangles == 0) {
        return;
    }

    cudaMalloc(&dev_triangles, totalNumberOfTriangles * sizeof(Triangle));
    int offset = 0;
    for (int i = 0; i < totalNumberOfGeom; i++) {
        if (geometries[i].type == MESH) {
            // Copy each geometry's triangles to the device memory
            cudaMemcpy(dev_triangles + offset, geometries[i].triangles, geometries[i].numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
            
            // Update the device pointer in the geometry struct to point to device memory
            geometries[i].devTriangles = dev_triangles + offset;
            
            // Move the offset by the number of triangles in this geometry
            offset += geometries[i].numTriangles;
        }
    }
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    int totalNumberOfGeom = scene->geoms.size();
    initialiseTriangles(dev_geomTriangles, scene->geoms, totalNumberOfGeom); // Must appear before initializing dev_geoms
    cudaMalloc(&dev_geoms, totalNumberOfGeom * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    int totalNumberOfLights = scene->lights.size();
    initialiseTriangles(dev_lightTriangles, scene->lights, totalNumberOfLights); // Must appear before initializing dev_lights
    cudaMalloc(&dev_lights, totalNumberOfLights * sizeof(Geom));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_totalNumberOfLights, sizeof(int));
    cudaMemcpy(dev_totalNumberOfLights, &totalNumberOfLights, sizeof(int), cudaMemcpyHostToDevice);

    // We've already got the triangles in the device memory, so we can delete them from the host memory
    delete scene->geoms.data()->triangles;

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_lights);
    cudaFree(dev_geomTriangles);
    cudaFree(dev_lightTriangles);
    cudaFree(dev_totalNumberOfLights);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }

    int index = x + (y * cam.resolution.x);

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
    thrust::uniform_real_distribution<float> u01(0, 1);

    PathSegment& segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f);

    glm::vec2 offset = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f), 0.5f * (u01(rng) * 2.0f - 1.0f));
    segment.ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + offset[0])
        - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + offset[1])
    );

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    segment.hasHitLight = false;
    segment.eta = 1.0f;
}

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
            else if (geom.type == MESH) {
                t = meshIntersectionTestNaive(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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

__global__ void shadeNaive(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials) {
    // As long as we enter here, it means the ray has remaining bounces > 0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];
    if (intersection.t <= 0.0f) {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];
    glm::vec3 materialColor = material.color;
    
    if (material.emittance > 0.0f) {
        pathSegment.color *= materialColor * material.emittance;
        pathSegment.remainingBounces = 0;
        pathSegment.hasHitLight = true;
    }
    else {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        glm::vec3 oldIntersect = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 surfaceNormal = glm::normalize(intersection.surfaceNormal);
        glm::vec3 woW = -pathSegment.ray.direction;
        glm::vec3 wiW;
        glm::vec3 c;
        float pdf;
        float eta;
        
        scatterRay(pathSegment, woW, surfaceNormal, wiW, pdf, c, eta, material, rng); 

        pathSegment.ray.direction = wiW; // wiW should already be normalized
        // Without the offset, when the ray immediately intersects the surface it originated from, the refraction calculations may fail or yield invalid results, such as:
        // Total Internal Reflection: The refracted ray might get treated as a reflective ray due to intersection problems, resulting in no transmitted light.
        // Black Pixels: The lack of refraction or valid light contribution can result in areas appearing black, as seen in your case.
        pathSegment.ray.origin = oldIntersect + pathSegment.ray.direction * 0.01f;
        pathSegment.color *= c; 
        
        #if (USE_RUSSIAN_ROULETTE) // Possibly terminate the path with Russian roulette
            if (depth > 3) {
                // So that the ray can bounce for a bit before we start terminating it
                float maxComponent = fmaxf(c.x, fmaxf(c.y, c.z));
                float survivalProbability = u01(rng);
                float eta_sq = eta * eta;
                float q = fminf(maxComponent * eta_sq, 0.99f);
                
                if (q < survivalProbability) {
                    pathSegment.remainingBounces = 0;
                    return;
                }
                else {
                    pathSegment.color /= q;
                }
            }        
        #endif
        
        pathSegment.remainingBounces--;
    }
}

__host__ __device__ int sampleTriangleFromMesh(const Triangle* triangles, const int numTriangles, const float randVal) {
    // Perform a binary search over the CDF to find the corresponding triangle
    int left = 0;
    int right = numTriangles - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (randVal < triangles[mid].cdf) {
            right = mid;
        }
        else {
            left = mid + 1;
        }
    }

    return left;
}

// Sample a light source and return the sampled point in world space
__host__ __device__ glm::vec3 sampleLight(const int totalNumberOfLights, const Geom* lights, const Material* mats, thrust::default_random_engine &rng, glm::vec3 &sampledPointWorld, glm::vec3 &sampledNormalWorld, float &pdf) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    // Randomly sample an emitter from the list, ensuring the index doesn't exceed totalLights - 1
    int light_idx = min(int(u01(rng) * totalNumberOfLights), totalNumberOfLights - 1);
    // Get the emitter from the list
    Geom light = lights[light_idx];

    // So far we assum only a mesh can be a light and it's an area light
    if (light.type == MESH) {
        glm::vec3 samples = glm::vec3(u01(rng), u01(rng), u01(rng));
        int triangleIdx = sampleTriangleFromMesh(light.devTriangles, light.numTriangles, samples.x);
        Triangle lightTriangle = light.devTriangles[triangleIdx];
        float alpha = 1.0f - sqrt(1.0f - samples.y);
        float beta = samples.z * sqrt(1.0f - samples.y);
        float gamma = 1.0f - alpha - beta;

        glm::vec3 sampledPointLocal = alpha * lightTriangle.points[0] + beta * lightTriangle.points[1] + gamma * lightTriangle.points[2];
        glm::vec3 sampledNormalLocal = glm::normalize(alpha * lightTriangle.normals[0] + beta * lightTriangle.normals[1] + gamma * lightTriangle.normals[2]);

        sampledPointWorld = multiplyMV(light.transform, glm::vec4(sampledPointLocal, 1.0f));
        sampledNormalWorld = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(sampledNormalLocal, 0.0f)));
        pdf = 1.0f / light.area / totalNumberOfLights;
        Material lightMaterial = mats[light.materialid];
        return lightMaterial.color * lightMaterial.emittance;
    }

    return glm::vec3(0.0f);
}

__host__ __device__ bool isRayOccluded(const int geomsSize, const Geom* geoms, Ray &ray) {
    float t;
    bool outside = true;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    for (int i = 0; i < geomsSize; i++) {
        Geom geom = geoms[i];

        if (geom.type == CUBE) {
            t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE) {
            t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == MESH) {
            t = meshIntersectionTestNaive(geom, ray, tmp_intersect, tmp_normal, outside);
        }

        if (t > 0.0f) {
            return true;
        }
    }

    return false;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.hasHitLight) {
            image[iterationPath.pixelIndex] += iterationPath.color;  
        }   
    }
}

__global__ void computeIsIntersected(int num_paths, int* isIntersected, const ShadeableIntersection* intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    isIntersected[idx] = intersections[idx].t != -1.0f;
}

void partitionRays(int &num_paths, PathSegment* dev_paths, const ShadeableIntersection* dev_intersections) {
    thrust::device_ptr<PathSegment> dev_ptr(dev_paths);
    thrust::device_ptr<PathSegment> dev_ptr_end = thrust::stable_partition(thrust::device, dev_ptr, dev_ptr + num_paths, has_remaining_bounces());
    cudaDeviceSynchronize();
    num_paths = dev_ptr_end - dev_ptr;
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int total_num_paths = dev_path_end - dev_paths;
    int num_paths = total_num_paths;

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
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // Sort materials by type
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortMaterialCondition());
        cudaDeviceSynchronize();

        shadeNaive<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        cudaDeviceSynchronize();

        // compact paths
        partitionRays(num_paths, dev_paths, dev_intersections);

        iterationComplete = (depth >= traceDepth) || (num_paths == 0);
        
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(total_num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}