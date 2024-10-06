#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

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
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* dev_image, glm::vec3* dev_denoiseImg, glm::vec3* dev_final_image,
    float percentDenoise)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float percentRegular = 1 - percentDenoise;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix1 = dev_image[index];
        glm::vec3 pix2 = dev_denoiseImg[index];
        glm::vec3 pix = percentRegular * pix1 + percentDenoise * pix2;

        dev_final_image[index] = pix;
        //pix = pix1;

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

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_denoiseImg = NULL;
static glm::vec3* dev_final_image = NULL;
static glm::vec3* dev_normalsImg = NULL;
static glm::vec3* dev_albedoImg = NULL;

static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static MeshTriangle* dev_triangleBuffer_0 = NULL;
//static Triangle* dev_triangleBuffer_1 = NULL;

static std::vector<cudaTextureObject_t> host_texObjs;
static std::vector<cudaArray_t> dev_cuArrays;
static cudaTextureObject_t* dev_textureObjIDs;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    //hst_scene->geoms.p
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoiseImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoiseImg, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_final_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_final_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normalsImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normalsImg, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedoImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedoImg, 1, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memory you need

    //Initialize Triangle Memory!
    std::vector<MeshTriangle> triangles = hst_scene->getTriangleBuffer();
    cudaMalloc(&dev_triangleBuffer_0, triangles.size() * sizeof(MeshTriangle));
    cudaMemcpy(dev_triangleBuffer_0, triangles.data(), triangles.size() * sizeof(MeshTriangle), cudaMemcpyHostToDevice);

    checkCUDAError("Triangle Buffer Init");

    //CUDA TEXTURE OBJECTS!
    //1 texture object for every unique texture
    std::vector<tinygltf::Image> images = hst_scene->getImages();
    for (const tinygltf::Image& image : images) {
        cudaChannelFormatKind formatType;
        cudaChannelFormatDesc channelDesc;
        if (image.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            formatType = cudaChannelFormatKindUnsigned;
            if (image.component == 3) {
                channelDesc = cudaCreateChannelDesc<uchar3>();
            }
            else {
                channelDesc = cudaCreateChannelDesc<uchar4>();
            }
        }
        else {
            formatType = cudaChannelFormatKindFloat;
            if (image.component == 3) {
                channelDesc = cudaCreateChannelDesc<float3>();
            }
            else {
                channelDesc = cudaCreateChannelDesc<float4>();
            }
        }
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, image.width, image.height);
        cudaMemcpy2DToArray(cuArray, 0, 0, 
            image.image.data(), 
            image.width * sizeof(float), 
            image.width * sizeof(float),
            image.height,
            cudaMemcpyHostToDevice);
        //checkCUDAError("aight");
        dev_cuArrays.push_back(cuArray);

        //// Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
        checkCUDAError("textureObject Init");
        host_texObjs.push_back(texObj);
    }

    cudaMalloc((void**)&dev_textureObjIDs, host_texObjs.size() * sizeof(cudaTextureObject_t));

    //for (int i = 0; i < host_texObjs.size(); i++) {
    //    std::cout << "dev_texObj val: " << host_texObjs[i] << "\n";
    //}
    // Copy texture objects from the host vector to the device
    cudaMemcpy(dev_textureObjIDs, host_texObjs.data(), host_texObjs.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_denoiseImg);
    cudaFree(dev_final_image);
    cudaFree(dev_normalsImg);
    cudaFree(dev_albedoImg);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_triangleBuffer_0);

    for (cudaArray_t cuArray : dev_cuArrays) {
        if (cuArray != nullptr) {
            cudaError_t err = cudaFreeArray(cuArray);
            if (err != cudaSuccess) {
                std::cerr << "Failed to free CUDA array: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
    dev_cuArrays.clear();

    for (int i = 0; i < host_texObjs.size(); i++) {
        cudaTextureObject_t texObj = host_texObjs[i];
        cudaError_t err = cudaDestroyTextureObject(texObj);
        if (err != cudaSuccess) {
            std::cerr << "Failed to destroy texture object: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDestroyTextureObject(host_texObjs[i]);
    }
    host_texObjs.clear();


    cudaFree(dev_textureObjIDs);


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
        segment.L = glm::vec3(0.0f, 0.0f, 0.0f); // Used to be (1.0, 1.0, 1.0)
        segment.beta = glm::vec3(1, 1, 1);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> uhalf(0.0, 0.5);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + 0.5f + uhalf(rng))
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + 0.5f + uhalf(rng))
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
    MeshTriangle* triangles,
    cudaTextureObject_t* texObjs,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    //Don't compute if the segment is already complete!
    if (path_index < num_paths && pathSegments[path_index].remainingBounces > 0)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 texCol;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec3 tmp_texCol;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_texCol = glm::vec3(-1, -1, -1);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_texCol = glm::vec3(-1, -1, -1);
            }
            else if (geom.type == TRI)
            {
                t = triangleIntersectionTest(geom, pathSegment.ray, triangles[geom.triangle_index], tmp_intersect, tmp_normal, outside);
                tmp_texCol = glm::vec3(1,0,1);
                //material.
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
                texCol = tmp_texCol;
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
            intersections[path_index].texCol = texCol;
        }
    }
}

/**
* Accumulate normals and albedo in their buffers
*/
__global__ void denoise_shade(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    glm::vec3* normalsImg,
    glm::vec3* albedoImg,
    int curItr,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        int pixelIndex = pathSegments[idx].pixelIndex;
        if (intersection.t > 0) { //intersection
            Material material = materials[intersection.materialId];
            normalsImg[pixelIndex] *= (curItr - 1);
            normalsImg[pixelIndex] += intersection.surfaceNormal;
            normalsImg[pixelIndex] /= curItr;
            if (material.emittance > 0) {
                glm::vec3 color = (intersection.texCol.x != -1) ? intersection.texCol : material.color;
                glm::vec3 a = glm::clamp(color * material.emittance, glm::vec3(0), glm::vec3(1));
                albedoImg[pixelIndex] *= (curItr - 1);
                albedoImg[pixelIndex] += a;
                albedoImg[pixelIndex] /= curItr;
            }
            else {
                glm::vec3 color = (intersection.texCol.x != -1) ? intersection.texCol : material.color;
                glm::vec3 a = glm::clamp(color, glm::vec3(0), glm::vec3(1));
                albedoImg[pixelIndex] *= (curItr - 1);
                albedoImg[pixelIndex] += a;
                albedoImg[pixelIndex] /= curItr;
            }
        }
        else { //no intersection
            float factor = ((curItr - 1) / curItr);
            normalsImg[pixelIndex] *= factor;
            albedoImg[pixelIndex] *= factor;
        }
    }
}
///  Iterative lighting logic:
///  LTE:
///  L_o = L_e + integral(f() * Li(w_i) * absdot)_dw_i
///  L_o = L_e + (f() * Li(w_i) * absdot) / pdf(w_i)
__global__ void naive_shade(int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        bool useTexCol = (intersection.texCol.x != -1);
        if (intersection.t > 0 && pathSegments[idx].remainingBounces > 0) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            Material material = materials[intersection.materialId];
            pathSegments[idx].ray.origin = getPointOnRay(pathSegments[idx].ray, intersection.t);
            pathSegments[idx].remainingBounces--;

            if (material.emittance > 0) {
                glm::vec3 color = useTexCol ? intersection.texCol : material.color;
                glm::vec3 Le = color * material.emittance;
                pathSegments[idx].L = pathSegments[idx].beta * Le;
                pathSegments[idx].remainingBounces = 0;
                return;
            }

            float pdf;
            glm::vec3 f;
            glm::vec3 woWOut = -pathSegments[idx].ray.direction;
            sample_f(pathSegments[idx], woWOut, pdf, f, intersection.surfaceNormal, material, intersection.texCol, useTexCol, rng);
            if (pdf < 0.00000001f || f == glm::vec3(0))
                return;

            float absdot = glm::abs(glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal));
            pathSegments[idx].beta *= f * absdot / pdf;
        }
        else {
            return;
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
__global__ void shadeFakeMaterial(
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
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].beta *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].beta *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].beta *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].beta = glm::vec3(0.0f); //Intersects with the void .. so, zero color
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, int cur_iter)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] *= (cur_iter - 1);
        image[iterationPath.pixelIndex] += iterationPath.L; //should be L, not beta
        image[iterationPath.pixelIndex] /= cur_iter;
    }
}

//__global__ void finalDivide()
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, oidn::FilterRef& oidn_filter, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    //const int traceDepth = 100;
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

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            dev_triangleBuffer_0,
            dev_textureObjIDs,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        //For every iteration, at the first intersection!
        if (depth == 1) {
            denoise_shade<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths,
                dev_intersections,
                dev_paths,
                dev_normalsImg,
                dev_albedoImg,
                iter,
                dev_materials
            );
        }
        

#if SORTBYMATERIAL == 1
        thrust::device_ptr<ShadeableIntersection> d_itr_ptr(dev_intersections);
        thrust::device_ptr<PathSegment> d_paths_ptr(dev_paths);
        thrust::device_vector<int> d_keys(num_paths);
        thrust::transform(d_itr_ptr, d_itr_ptr + num_paths, d_keys.begin(), getMatId());

        //sort both d_itr_ptr and d_paths_ptr based on the sorting of the materialID buffer
        thrust::sort_by_key(d_keys.begin(), d_keys.begin() + num_paths, 
            thrust::make_zip_iterator(thrust::make_tuple(d_itr_ptr, d_paths_ptr)));
#endif
        naive_shade<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        checkCUDAError("shade 1 depth of path segments");

#if STREAMCOMPACTION == 1
        thrust::device_ptr<PathSegment> d_ptr(dev_paths);

        auto new_end = thrust::partition(d_ptr, d_ptr + num_paths, CheckRemainingBounces());

        num_paths = thrust::distance(d_ptr, new_end);
#endif

        if (num_paths == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths, iter);
    checkCUDAError("finalGather step on beauty pass (dev_image)");

    // Run denoising!

    if (iter % 3 == 0) {
        oidn_filter.setImage("color", dev_image, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
        oidn_filter.setImage("albedo", dev_albedoImg, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
        oidn_filter.setImage("normal", dev_normalsImg, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
        oidn_filter.setImage("output", dev_denoiseImg, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);

        oidn_filter.commit();
        oidn_filter.execute();
    }

    ///////////////////////////////////////////////////////////////////////////
    // Send results to OpenGL buffer for rendering
    // Modify this to send dev_denoiseImg instead of dev_image!

    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image, dev_denoiseImg, dev_final_image, 0.9);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_final_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
