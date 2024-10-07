#include "pathtrace.h"
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "device_launch_parameters.h"

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

//  Set seed for thrust random number generator
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Thrust sort materials comparator
struct CompInterMats {
    __host__ __device__ bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2) {
        return s1.materialId <= s2.materialId;
    }
};

// Thrust compact terminated rays conditioner
struct CheckRayBounce {
    __host__ __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
    }
};

// Kernel that writes the image to the OpenGL PBO directly
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

// Device variables
// Render components
static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;

// Scene components
static Geom* dev_geoms = NULL;
static Primitive* dev_primitives = NULL;
static BVHNode* dev_bvhNodes = NULL;
static BSDF* dev_materials = NULL;
static Light* dev_lights = NULL;

// Compute components
static glm::vec3* dev_image = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

//__global__ void testCopiedBVH(BVHNode* dev_bvhNodes)
//{
//    for (int i = 0; i < 11; i++) {
//        BVHNode& b = dev_bvhNodes[i];
//
//        BBox& bb = b.bb;
//
//        float mx = bb.minC.x;
//        float my = bb.minC.y;
//        float mz = bb.minC.z;
//
//        float mxx = bb.maxC[0];
//        float mxy = bb.maxC[1];
//        float mxz = bb.maxC[2];
//
//        int l = b.leftNodeIndex;
//        int r = b.rightNodeIndex;
//        int pN = b.numPrims;
//
//        bool il = b.isLeaf();
//
//        mx++;
//        if (mx > 0 && mxx > 0) {
//            for (int j = 0; j < pN; j++) {
//                int curPrimI = b.primsIndices[j];
//            }
//
//        }
//    }
//}

void pathtraceInit(Scene* scene)
{
    // Render
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // Scene components
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_primitives, scene->prims.size() * sizeof(Primitive));
    cudaMemcpy(dev_primitives, scene->prims.data(), scene->prims.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    
#if BVH
    cudaMalloc(&dev_bvhNodes, scene->bvh.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvh.data(), scene->bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    /*for (int i = 0; i < scene->bvh.size(); i++)
    {
        if (scene->bvh[i].numPrims > 0)
        {
            int* d_primitiveIndices = nullptr;
            cudaMalloc((void**)&d_primitiveIndices, scene->bvh[i].numPrims * sizeof(int));
            cudaMemcpy(d_primitiveIndices, scene->bvh[i].primsIndices, scene->bvh[i].numPrims * sizeof(int), cudaMemcpyHostToDevice);

            cudaMemcpy(&(dev_bvhNodes[i].primsIndices), &d_primitiveIndices, sizeof(int*), cudaMemcpyHostToDevice);
        }
    }*/

    //copyBVHNodes(scene->bvh, dev_bvhNodes);
    //testCopiedBVH << <1, 1 >> > (dev_bvhNodes);
#endif

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(BSDF));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(BSDF), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    // Compute components
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

//void copyBVHNodes(std::vector<BVHNode>& bvh, BVHNode* dev_bvhNodes) 
//{
//    cudaMalloc(&dev_bvhNodes, bvh.size() * sizeof(BVHNode));
//    cudaMemcpy(dev_bvhNodes, bvh.data(), bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
//
//    for (int i = 0; i < bvh.size(); i++)
//    {
//        if (bvh[i].numPrims > 0) 
//        {
//            int* d_primitiveIndices = nullptr;
//            cudaMalloc((void**)&d_primitiveIndices, bvh[i].numPrims * sizeof(int));
//            cudaMemcpy(d_primitiveIndices, bvh[i].primsIndices, bvh[i].numPrims * sizeof(int), cudaMemcpyHostToDevice);
//
//            cudaMemcpy(&(dev_bvhNodes[i].primsIndices), &d_primitiveIndices, sizeof(int*), cudaMemcpyHostToDevice);
//        }
//    }
//}

//void freeBVHNode(BVHNode* dev_bvhNodes) 
//{
//    if (hst_scene != NULL && dev_bvhNodes != NULL)
//    {
//        std::cout << (hst_scene->bvh).size() << " test?" << std::endl;
//        for (int i = 0; i < (hst_scene->bvh).size(); i++)
//        {
//            if (dev_bvhNodes[i].numPrims > 0)
//            {
//                cudaFree(dev_bvhNodes[i].primsIndices);
//            }
//        }
//    }
//
//    cudaFree(dev_bvhNodes);
//}

void pathtraceFree()
{
    cudaFree(dev_geoms);
    cudaFree(dev_primitives);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_materials);
    cudaFree(dev_lights);
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_intersections);

    checkCUDAError("pathtraceFree");
}

// Concentric disk Sampling on thin lens
__device__ glm::vec2 concentricDiskSampling(thrust::default_random_engine rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float u1 = u01(rng);
    float u2 = u01(rng);

    float r, theta;
    //  Center of the disk
    if (u1 == 0 && u2 == 0) {
        return glm::vec2(0, 0);
    }

    // Map square to disk
    if (abs(u1) > abs(u2)) {
        r = u1;
        theta = (PI / 4) * (u2 / u1);
    }
    else {
        r = u2;
        theta = (PI / 2) - (PI / 4) * (u1 / u2);
    }

    // Convert to Cartesian coordinates
    return glm::vec2(r * cos(theta), r * sin(theta));
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
        segment.ray.tmin = 0.0f;
        segment.ray.tmax = 1e38f;
        segment.L_out = glm::vec3(0.0f);
        segment.beta = glm::vec3(1.0f);

        // Stochastic Antialiasing
#if StochasticAntialiasing
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> uh(-0.5, 0.5);
        float jitterX = uh(rng);
        float jitterY = uh(rng);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

#if DOF
        glm::vec2 lensSample = cam.aperture * concentricDiskSampling(rng);
        glm::vec3 lensPoint = cam.position + lensSample.x * cam.right + lensSample.y * cam.up;
        glm::vec3 focalPoint = cam.position + segment.ray.direction * cam.focal;
        glm::vec3 newDir = glm::normalize(focalPoint - lensPoint);
        segment.ray.origin = lensPoint;
        segment.ray.direction = newDir;
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    Primitive* prims,
    int prims_size,
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

        for (int i = 0; i < prims_size; i++)
        {
            Primitive& prim = prims[i];

            if (prim.type == CUBEP)
            {
                t = boxIntersectionTest(geoms[prim.geomId], pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (prim.type == SPHEREP)
            {
                t = sphereIntersectionTest(geoms[prim.geomId], pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (prim.type == TRIANGLE)
            {
                t = triangleIntersectionTest(geoms[prim.geomId], prim, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else
            {
                t = -1;
            }

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
            intersections[path_index].materialId = geoms[prims[hit_geom_index].geomId].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__global__ void computeIntersectionBVH(
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    Primitive* prims,
    ShadeableIntersection* intersections,
    BVHNode* bvh)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment& pathSegment = pathSegments[path_index];
        ShadeableIntersection& intersection = intersections[path_index];
        intersection.t = -1.f;
        intersectBVH(pathSegment.ray, intersection, geoms, prims, bvh);
    }
}

__global__ void oneBounceRadiance(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    BSDF* materials) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        
        ShadeableIntersection& intersection = shadeableIntersections[idx];
        PathSegment& pathSegment = pathSegments[idx];

        if (intersection.t > 0.0f) {

            BSDF material = materials[intersection.materialId];

            if (material.getType() == EMISSION) {
                pathSegment.L_out += pathSegment.beta * material.getEmission();
                pathSegment.remainingBounces = 0;
            }
            else {
                glm::mat3 o2w;
                makeCoordTrans(o2w, intersection.surfaceNormal);
                glm::mat3 w2o = glm::transpose(o2w);

                glm::vec3 hit = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                glm::vec3 wo = w2o * (-pathSegment.ray.direction);

                glm::vec3 wi;
                float pdf;
                
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

                glm::vec3 sp = material.sampleF(wo, wi, &pdf, rng);
                
                glm::vec3 w_world = glm::normalize(o2w * wi);
                pathSegment.beta *= (sp * cosThetaUnit(wi) / pdf);
                pathSegment.ray.origin = EPSILON * w_world + hit;
                pathSegment.ray.direction = w_world;
                pathSegment.ray.tmax = 1e38f;
#if RussianRoulette
                float continueP = glm::clamp(glm::max(glm::max(pathSegment.beta.x, pathSegment.beta.y), pathSegment.beta.z), 0.0f, 1.0f);
                thrust::uniform_real_distribution<float> u01(0, 1);
                float terminateP = u01(rng);
                if (terminateP > continueP) 
                {
                    pathSegment.remainingBounces = 0;
                }
                else 
                {
                    pathSegment.beta /= continueP;
                }
#else
                pathSegment.remainingBounces -= 1;
#endif
            }
        }
        else {
            pathSegment.remainingBounces = 0;
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
//__global__ void shadeFakeMaterial(
//    int iter,
//    int num_paths,
//    ShadeableIntersection* shadeableIntersections,
//    PathSegment* pathSegments,
//    Material* materials)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < num_paths)
//    {
//        ShadeableIntersection intersection = shadeableIntersections[idx];
//
//        if (pathSegments[idx].remainingBounces == 0) return;
//
//        if (intersection.t > 0.0f) // if the intersection exists...
//        {
//          // Set up the RNG
//          // LOOK: this is how you use thrust's RNG! Please look at
//          // makeSeededRandomEngine as well.
//            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
//            thrust::uniform_real_distribution<float> u01(0, 1);
//
//            Material material = materials[intersection.materialId];
//            glm::vec3 materialColor = material.color;
//
//            // If the material indicates that the object was a light, "light" the ray
//            if (material.emittance > 0.0f) {
//                pathSegments[idx].color *= material.color * material.emittance;
//                pathSegments[idx].remainingBounces = 0;
//            }
//            // Otherwise, do some pseudo-lighting computation. This is actually more
//            // like what you would expect from shading in a rasterizer like OpenGL.
//            // TODO: replace this! you should be able to start with basically a one-liner
//            else {
//                scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction, intersection.surfaceNormal, material, rng);
//            }
//            // If there was no intersection, color the ray black.
//            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
//            // used for opacity, in which case they can indicate "no opacity".
//            // This can be useful for post-processing and image compositing.
//        }
//        else {
//            pathSegments[idx].color = glm::vec3(0.0f);
//            pathSegments[idx].remainingBounces = 0;
//        }
//    }
//}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.L_out;
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

    // Generate jittered rays from the camera into the screen
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // Ready to traverse
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

#if BVH
        computeIntersectionBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths,
            dev_paths,
            dev_geoms,
            dev_primitives,
            dev_intersections,
            dev_bvhNodes
            );
#else
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            num_paths,
            dev_paths,
            dev_geoms,
            dev_primitives,
            hst_scene->prims.size(),
            dev_intersections
        );
#endif

        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // sort by intersection materials
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, CompInterMats());

        // naive shading
        oneBounceRadiance<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        checkCUDAError("one bounce shading");
        cudaDeviceSynchronize();

        // compact terminated rays
        PathSegment* path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, CheckRayBounce());
        num_paths = path_end - dev_paths;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
        depth++;

        if (num_paths == 0 || traceDepth == depth) iterationComplete = true;
    }

    // Recover total number path segments
    num_paths = dev_path_end - dev_paths;
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
