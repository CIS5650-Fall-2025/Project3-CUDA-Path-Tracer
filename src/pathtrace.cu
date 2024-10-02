#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"

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
// TODO: static variables for device memory, any extra info you need, etc
// ...
static glm::vec3* dev_meshVertices = NULL;
static glm::ivec3* dev_meshFaceIndices = NULL;
static glm::vec3* dev_meshFaceNormals = NULL;
static int* dev_meshFaceMatIndices = NULL;

static bvhNode* dev_bvhNodes = NULL;
static int* dev_meshFaceIndicesBVH = NULL;

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
    
    // mesh 
    cudaMalloc(&dev_meshVertices, scene->mesh.vertices.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_meshVertices, scene->mesh.vertices.data(), scene->mesh.vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_meshFaceIndices, scene->mesh.faceIndices.size() * sizeof(glm::ivec3));
    cudaMemcpy(dev_meshFaceIndices, scene->mesh.faceIndices.data(), scene->mesh.faceIndices.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_meshFaceNormals, scene->mesh.faceNormals.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_meshFaceNormals, scene->mesh.faceNormals.data(), scene->mesh.faceNormals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_meshFaceMatIndices, scene->mesh.faceMatIndices.size() * sizeof(int));
    cudaMemcpy(dev_meshFaceMatIndices, scene->mesh.faceMatIndices.data(), scene->mesh.faceMatIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    // json geom
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    // material
    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    // bvh
    cudaMalloc(&dev_bvhNodes, sizeof(bvhNode) * scene->bvhNodes.size());
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(bvhNode), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_meshFaceIndicesBVH, scene->mesh.faceIndicesBVH.size() * sizeof(int));
    cudaMemcpy(dev_meshFaceIndicesBVH, scene->mesh.faceIndicesBVH.data(), scene->mesh.faceIndicesBVH.size() * sizeof(int), cudaMemcpyHostToDevice);
    
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
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_meshVertices);
    cudaFree(dev_meshFaceIndices);
    cudaFree(dev_meshFaceNormals);
    cudaFree(dev_meshFaceMatIndices);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_meshFaceIndicesBVH);
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

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
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
    glm::vec3* vertices,
    glm::ivec3* faceIndices,
    glm::vec3* faceNormals,
    int* faceMatIndices,
    bvhNode* bvhNodes,
    int* faceIndicesBVH,
    int face_count,
    ShadeableIntersection* intersections,
    bool useBVH)
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
        int material_id = -1;

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
                material_id = geom.materialid;
            }
        }
        Ray &ray = pathSegment.ray;
        if (!useBVH){
            for (int i = 0; i < face_count; i++) {
                glm::ivec3& face = faceIndices[i];
                glm::vec3& v0 = vertices[face.x];
                glm::vec3& v1 = vertices[face.y];
                glm::vec3& v2 = vertices[face.z];

                glm::vec3 baryPosition;
                if (glm::intersectRayTriangle(ray.origin, ray.direction, v0, v1, v2, baryPosition)) {
                    t = baryPosition.z;
                    // Check if the ray direction is in the opposite direction as the normal
                    glm::vec3 faceNormal = faceNormals[i];
                    if (t > 0.0f && t < t_min) {
                        t_min = t;
                        hit_geom_index = i;
                        intersect_point = ray.origin + t * ray.direction;
                        normal = faceNormals[i];
                        material_id = faceMatIndices[i];
                    }
                }
            }
        }else{
            bool hit;
            float t = FLT_MAX;
            int faceIndexHit = -1;
            BVHHitTestIterative(
                ray, bvhNodes, 
                vertices, faceIndices, faceNormals, faceIndicesBVH,
                t, faceIndexHit, hit);
            if (t > 0.0f && t < t_min) {
                t_min = t;
                hit_geom_index = faceIndexHit;
                normal = faceNormals[faceIndexHit];
                material_id = faceMatIndices[faceIndexHit];
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            pathSegments[path_index].remainingBounces = 0;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = material_id;
            intersections[path_index].surfaceNormal = normal;
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
    int currDepth,
    int maxDepth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    bool renderWithPathTracing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (pathSegments[idx].remainingBounces <= 0) {
                return;
            }
            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }else if (currDepth < maxDepth) {
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
                if (renderWithPathTracing){
                    scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
                    pathSegments[idx].remainingBounces--;
                }else{
                    float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                    pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                    pathSegments[idx].color *= u01(rng); // apply some noise because why not
                    pathSegments[idx].remainingBounces = 0;
                }
            }
            else {
                pathSegments[idx].color = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = 0;
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

struct CompactPathSegments {
    __host__ __device__
    bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
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
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // collision check
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_meshVertices,
            dev_meshFaceIndices,
            dev_meshFaceNormals,
            dev_meshFaceMatIndices,
            dev_bvhNodes,
            dev_meshFaceIndicesBVH,
            hst_scene->mesh.faceIndices.size(),
            dev_intersections,
            hst_scene->useBVH || hst_scene->useBasicBVC
        );
        checkCUDAError("computeIntersections");
        cudaDeviceSynchronize();
        depth++;

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
            depth,
            traceDepth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            hst_scene->renderWithPathTracing
        );
        checkCUDAError("shadeMaterial");
        
        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, CompactPathSegments());
        num_paths = dev_path_end - dev_paths;
        iterationComplete = (depth >= traceDepth || num_paths <= 0);
        checkCUDAError("stream compact");

        if (guiData != NULL)
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
