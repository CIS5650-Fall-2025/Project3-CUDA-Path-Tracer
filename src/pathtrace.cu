#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "texture_utils.h"
#include "scene.h"
#include "denoise.h"


#define ERRORCHECK 1
#define ANTIALIASING 1
#define RAY_COMPACTION 1
#define MATERIAL_SORTING 0
#define DENOISE 1
#define BVH 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)


const int denoiseFrequency = 20;
const int min_denoise_iter = 5;


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
static glm::vec3* dev_denoised_image = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static std::vector<Triangle*> dev_mesh_triangles;
static std::vector<BVHNode*> dev_mesh_BVHNodes;


int* dev_materialIds;


// Create a single global denoiser instance
DenoiserState globalDenoiser;
static int lastDenoisedIter = 0;


thrust::device_ptr<int> dev_thrust_materialIds;
thrust::device_ptr<PathSegment> dev_thrust_paths;
thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections;

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

    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    Geom* host_geoms = new Geom[scene->geoms.size()];
    memcpy(host_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom));
    for (int i = 0; i < scene->geoms.size(); i++) {
        if (host_geoms[i].type == MESH) {
            int numTriangles = host_geoms[i].num_triangles;
            Triangle* dev_triangles;
            cudaMalloc(&dev_triangles, numTriangles * sizeof(Triangle));
            cudaMemcpy(dev_triangles, host_geoms[i].triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
            host_geoms[i].triangles = dev_triangles;
            dev_mesh_triangles.push_back(dev_triangles);

            int numBVHNodes = host_geoms[i].num_BVHNodes;
            BVHNode* dev_BVH;
            cudaMalloc(&dev_BVH, numBVHNodes * sizeof(BVHNode));
            cudaMemcpy(dev_BVH, host_geoms[i].bvhNodes, numBVHNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
            host_geoms[i].bvhNodes = dev_BVH;
            dev_mesh_BVHNodes.push_back(dev_BVH);
        }
    }

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, host_geoms, scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    delete[] host_geoms;

    for (Material& material : scene->materials)
    {
        if (material.albedoMapData.data != nullptr)
        {
            if (!createCudaTexture(material.albedoMapData, material.albedoMapTex)) {
                std::cerr << "Failed to create CUDA texture for albedo map." << std::endl;
                
                exit(EXIT_FAILURE);
            }
        }

        if (material.normalMapData.data != nullptr)
        {
            if (!createCudaTexture(material.normalMapData, material.normalMapTex)) {
                std::cerr << "Failed to create CUDA texture for normal map." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        if (material.envMapData.data != nullptr)
        {
            if (!createCudaTexture(material.envMapData, material.envMap)) {
                std::cerr << "Failed to create CUDA texture for env map." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

    }

    for (Material& material : scene->materials)
    {
        material.albedoMapData.data = nullptr;
        material.normalMapData.data = nullptr;
    }

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));
    cudaMemset(dev_materialIds, 0, pixelcount * sizeof(int));

    setupOIDN(globalDenoiser, cam.resolution.x, cam.resolution.y);

    checkCUDAError("pathtraceInit");
}


void pathtraceFree(Scene* scene)
{
    
    //all used for texture debugging, remember to allocate the texture obj after cuda initialization, 
    // so holding texture data using a host pointer is correct way
    cudaError_t err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error synchronizing device: " << cudaGetErrorString(err) << std::endl;
    }


    for (Triangle* dev_triangles : dev_mesh_triangles) {
        err = cudaFree(dev_triangles);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_triangles: " << cudaGetErrorString(err) << std::endl;
        }
    }
    for (BVHNode* dev_bvh : dev_mesh_BVHNodes) {
        err = cudaFree(dev_bvh);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_bvh: " << cudaGetErrorString(err) << std::endl;
        }
    }
    dev_mesh_triangles.clear();
    dev_mesh_BVHNodes.clear();

    
    
    if (dev_image)
    {
        err = cudaFree(dev_image);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_image: " << cudaGetErrorString(err) << std::endl;
        }
        dev_image = nullptr;
    }  

    if (dev_paths)
    {
        err = cudaFree(dev_paths);;
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_paths: " << cudaGetErrorString(err) << std::endl;
        }
        dev_paths = nullptr;
    }
    if (dev_geoms)
    {
        err = cudaFree(dev_geoms);;
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_geoms: " << cudaGetErrorString(err) << std::endl;
        }
        dev_geoms = nullptr;
    }

    
    if (dev_materials)
    {
        err = cudaFree(dev_materials);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_materials: " << cudaGetErrorString(err) << std::endl;
        }
        dev_materials = nullptr;
    }
    if (dev_intersections)
    {
        err = cudaFree(dev_intersections);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_intersections: " << cudaGetErrorString(err) << std::endl;
        }
        dev_intersections = nullptr;
    }

    
    if (dev_denoised_image)
    {
        err = cudaFree(dev_denoised_image);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_denoised_image: " << cudaGetErrorString(err) << std::endl;
        }
        dev_denoised_image = nullptr;
    }

    // Free dev_albedo
    if (dev_albedo)
    {
        err = cudaFree(dev_albedo);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_albedo: " << cudaGetErrorString(err) << std::endl;
        }
        dev_albedo = nullptr;
    }

    // Free dev_normal
    if (dev_normal)
    {
        err = cudaFree(dev_normal);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_normal: " << cudaGetErrorString(err) << std::endl;
        }
        dev_normal = nullptr;
    }
    

    // TODO: clean up any extra device memory you created
    if(dev_materialIds)
    {
        err = cudaFree(dev_materialIds);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_intersections: " << cudaGetErrorString(err) << std::endl;
        }
        dev_materialIds = nullptr;
    }
    

    cleanupOIDN(globalDenoiser);

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
    //Each CUDA thread corresponds to one pixel.
    int x = (blockIdx.x * blockDim.x) + threadIdx.x; //pixel coord
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

#if ANTIALIASING
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // antialiasing by jittering the ray
        // Set up RNG per pixel
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Generate subpixel jitter
        float jitter_x = u01(rng);
        float jitter_y = u01(rng);
#else
        float jitter_x = 0.0f;
        float jitter_y = 0.0f;
#endif

        //This builds a ray going from the camera through the screen into the scene.
        //ray direction=camera forward+horizontal offset+vertical offset
        // - cam.resolution for centering around the camera's view
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitter_x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitter_y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

    }
}


__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections
    )
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
        glm::vec2 uv;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tempUV;

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
#if BVH
                t = meshIntersectionTest_WithMeshBVH(geom, pathSegment.ray, tmp_intersect, tmp_normal, tempUV, outside);
#else
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tempUV, outside);
#endif   
            }


            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tempUV;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].uv = glm::vec2(0.0f, 0.0f);
        }
        else
        {
            //Hit
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
        }
    }
}




__device__ glm::vec3 sampleAlbedoTexture(const Material& mat, glm::vec2 uv) {
    if (mat.albedoMapTex.texObj == 0) return glm::vec3(1.0f);

    float u = glm::clamp(uv.x, 0.0f, 1.0f);
    float v = glm::clamp(uv.y, 0.0f, 1.0f);
    float4 texColor = tex2D<float4>(mat.albedoMapTex.texObj, u, v);

    return glm::vec3(texColor.x, texColor.y, texColor.z);
}

__device__ glm::vec3 sampleNormalMap(const Material& mat, glm::vec2 uv, glm::vec3 fallback) {
    if (mat.normalMapTex.texObj == 0) return fallback;

    float u = glm::clamp(uv.x, 0.0f, 1.0f);
    float v = glm::clamp(uv.y, 0.0f, 1.0f);
    float4 texNormal = tex2D<float4>(mat.normalMapTex.texObj, u, v);
    glm::vec3 n(texNormal.x * 2.0f - 1.0f, texNormal.y * 2.0f - 1.0f, texNormal.z * 2.0f - 1.0f);
    return glm::normalize(n);
}

__device__ glm::vec3 sampleEnvironment(cudaTextureObject_t envMap, glm::vec3 dir, float intensity) {
    float theta = acosf(dir.y);
    float phi = atan2f(dir.z, dir.x);
    float u = (phi + PI) / (2.0f * PI);
    float v = theta / PI;
    float4 envColor = tex2D<float4>(envMap, u, v);
    return glm::vec3(envColor.x, envColor.y, envColor.z) * intensity;
}

__device__ void handleEnvironmentMiss(PathSegment& seg, PathPayload& payload,
    cudaTextureObject_t envMap, float envMapIntensity) {
    glm::vec3 rayDir = glm::normalize(seg.ray.direction);
    glm::vec3 envLight = (envMap != 0)
        ? sampleEnvironment(envMap, rayDir, envMapIntensity)
        : glm::vec3(0.0f);

    seg.color = glm::any(glm::lessThan(seg.color, glm::vec3(1.0f)))
        ? seg.color * envLight
        : envLight;

    payload.recordFirstBounce(glm::vec3(0.0f), envLight);
    seg.remainingBounces = 0;
    seg.ray.origin += 0.01f * seg.ray.direction;
}

__global__ void shadeMaterial(
    int iter,
    int numPaths,
    ShadeableIntersection* intersections,
    PathSegment* segments,
    Material* materials,
    int depth,
    glm::vec3* normals,
    glm::vec3* albedos,
    cudaTextureObject_t envMap,
    float envMapIntensity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    ShadeableIntersection isect = intersections[idx];
    PathSegment& ray = segments[idx];
    Material mat = materials[isect.materialId];

    PathPayload payload;
    payload.path = &ray;
    payload.intersection = isect;
    payload.material = mat;
    payload.pixelIdx = idx;
    payload.bounceDepth = depth;
    payload.gNormalBuffer = normals;
    payload.gAlbedoBuffer = albedos;

    if (isect.t > 0.0f) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

        glm::vec2 uv = isect.uv;
        glm::vec3 texColor = sampleAlbedoTexture(mat, uv);

        mat.color *= texColor;

        if (mat.hasReflective > 0.0f || mat.hasRefractive > 0.0f) {
            mat.specular.color *= texColor;
        }

        glm::vec3 normal = sampleNormalMap(mat, uv, isect.surfaceNormal);
        payload.material = mat;
        payload.intersection.surfaceNormal = normal;

        scatterRay(payload, rng);
    }
    else {
        handleEnvironmentMiss(ray, payload, envMap, envMapIntensity);
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





struct compareMaterialID
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& m_1, const ShadeableIntersection& m_2) const
    {
        return m_1.materialId < m_2.materialId;

    }
};


// stable_partition accepts functor object
struct isRayOngoing
{
    __host__ __device__
        bool operator()(const PathSegment& path) const
    {
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

    cudaTextureObject_t envMapTex = 0;
    float intensity = 1.0;

    // Find the environment material
    for (int i = 0; i < hst_scene->materials.size(); ++i) {
        Material& material = hst_scene->materials[i];

        if (material.is_env) {
            envMapTex = material.envMap.texObj;
            intensity = material.envMap_intensity;
            break;
        }
    }

   
    ///////////////////////////////////////////////////////////////////////////

    // === Generate primary rays ===
    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d>>> (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;



    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // === Reset intersections ===
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // === Intersect rays with scene geometry ===
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>>(
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


#if MATERIAL_SORTING
        //sort ray paths by material types
        auto dev_intersections_ptr = thrust::device_pointer_cast(dev_intersections); // thrust::device_ptr<ShadeableIntersection>
        auto dev_paths_materialSort_ptr = thrust::device_pointer_cast(dev_paths); //thrust::device_ptr<PathSegment>

        //Since CUDA kernels are asynchronous, make sure to synchronize the device before the thrust::stable_sort_by_key call
        cudaDeviceSynchronize();
        checkCUDAError("thrust::stable_sort_by_key");

        // stable_sort_by_key sorts one array (the keys) while rearranging a second array (the values) in tandem based on the sorting order of the first
        thrust::stable_sort_by_key(dev_intersections_ptr, dev_intersections_ptr + num_paths, dev_paths_materialSort_ptr, compareMaterialID());

        cudaDeviceSynchronize();
#endif
        

        shadeMaterial <<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth,
            dev_normal,
            dev_albedo,
            envMapTex,
            intensity
        );
        checkCUDAError("shadeMaterial");
        cudaDeviceSynchronize();



#if RAY_COMPACTION
        // keep only the rays (or paths) that still need further processing (i.e., those that have remaining bounces).
        // converts the raw device pointer into a Thrust device pointer type
        auto dev_paths_rayCompaction_ptr = thrust::device_pointer_cast(dev_paths);

        //reorders the elements in the range [dev_paths_ptr, dev_paths_ptr + num_paths) such that all elements for which the predicate isRayOngoing
        //returns true are placed before those for which it returns false.
        //The return value of thrust::stable_partition is an iterator (pointer) to the end of the partitioned section containing the active paths.
        auto dev_new_paths_ptr = thrust::stable_partition(
            thrust::device,
            dev_paths_rayCompaction_ptr,
            dev_paths_rayCompaction_ptr + num_paths,
            isRayOngoing()
        );
        num_paths = dev_new_paths_ptr - dev_paths_rayCompaction_ptr;
#endif


        bool maxDepthReached = (depth >= traceDepth);
        bool noActivePaths = num_paths <= 0;
        iterationComplete = maxDepthReached || noActivePaths;


        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
    


    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);


#if DENOISE
    if (iter >= min_denoise_iter)
    {
        if (iter == min_denoise_iter || iter % denoiseFrequency == 0) {
            runDenoisingPipeline(
                globalDenoiser,
                dev_image,
                dev_normal,
                dev_albedo,
                pixelcount,
                iter,
                dev_denoised_image);

            lastDenoisedIter = iter;
        }
    }
#endif
    
    // === Send result to OpenGL for display ===
#if DENOISE
    if (iter >= min_denoise_iter) {
        sendImageToPBO <<<blocksPerGrid2d, blockSize2d>>> (pbo, cam.resolution, lastDenoisedIter, dev_denoised_image);
    }
    else {
        sendImageToPBO <<<blocksPerGrid2d, blockSize2d>>> (pbo, cam.resolution, iter, dev_image); 
    }

#else
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
#endif


    // ===  Copy result back to host for saving ===
#if DENOISE
    cudaMemcpy(hst_scene->state.image.data(), dev_denoised_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif


    checkCUDAError("pathtrace");
}
