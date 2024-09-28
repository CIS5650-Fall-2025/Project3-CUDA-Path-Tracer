#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

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
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhnodes = NULL;
static glm::vec4* dev_tex_data = NULL;
static glm::vec4* dev_bumpmap_data = NULL;
static int* dev_tex_starts = NULL;
static int* dev_bump_starts = NULL;
static glm::vec2* dev_tex_dims = NULL;
static glm::vec2* dev_bump_dims = NULL;

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

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (scene->meshes.size() > 0) {
        int num_tris = scene->triangle_count;
        cudaMalloc(&dev_triangles, num_tris * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->mesh_triangles.data(), num_tris * sizeof(Triangle), cudaMemcpyHostToDevice);

        int num_bvhnodes = scene->bvhNodes.size();
        cudaMalloc(&dev_bvhnodes, num_bvhnodes * sizeof(BVHNode));
        cudaMemcpy(dev_bvhnodes, scene->bvhNodes.data(), num_bvhnodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }

    if (scene->textures.size() > 0) {
        int num_colors = 0;
        std::vector<glm::vec4> all_colors;
        for (Texture& tex : scene->textures) {
            num_colors += tex.color_data.size();
            all_colors.insert(all_colors.end(), tex.color_data.begin(), tex.color_data.end());
        }
        cudaMalloc(&dev_tex_data, num_colors * sizeof(glm::vec4));
        cudaMemcpy(dev_tex_data, all_colors.data(), num_colors * sizeof(glm::vec4), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_tex_starts, scene->tex_starts.size() * sizeof(int));
        cudaMemcpy(dev_tex_starts, scene->tex_starts.data(), scene->tex_starts.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_tex_dims, scene->tex_dims.size() * sizeof(glm::vec2));
        cudaMemcpy(dev_tex_dims, scene->tex_dims.data(), scene->tex_dims.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

        if (scene->bumpmaps.size() > 0) {
            int num_normals = 0;
            std::vector<glm::vec4> all_normals;
            for (Texture& tex : scene->bumpmaps) {
                num_colors += tex.color_data.size();
                all_normals.insert(all_normals.end(), tex.color_data.begin(), tex.color_data.end());
            }
            cudaMalloc(&dev_bumpmap_data, num_normals * sizeof(glm::vec4));
            cudaMemcpy(dev_bumpmap_data, all_normals.data(), num_normals * sizeof(glm::vec4), cudaMemcpyHostToDevice);

            cudaMalloc(&dev_bump_starts, scene->bump_starts.size() * sizeof(int));
            cudaMemcpy(dev_bump_starts, scene->bump_starts.data(), scene->bump_starts.size() * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc(&dev_bump_dims, scene->bump_dims.size() * sizeof(glm::vec2));
            cudaMemcpy(dev_bump_dims, scene->bump_dims.data(), scene->bump_dims.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
        }
    }
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
    cudaFree(dev_bvhnodes);
    cudaFree(dev_tex_data);
    cudaFree(dev_bumpmap_data);
    cudaFree(dev_tex_starts);
    cudaFree(dev_bump_starts);
    cudaFree(dev_tex_dims);
    cudaFree(dev_bump_dims);

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

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
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
    ShadeableIntersection* intersections,
    Triangle* tris,
    int num_tris,
    BVHNode* bvhnodes)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        glm::vec3 tmp_tangent;

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
                intersections->outside = outside;
            }
            else if (geom.type == TRIANGLE) {
                //t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH) {
#define USE_BVH 1
#if USE_BVH
                t = bvhIntersectionTest(pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside, bvhnodes, tris, num_tris);

#else
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tris, num_tris, tri_hit);
#endif
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
                tangent = tmp_tangent;
            }
        }

        intersections[path_index].outside = outside;

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
            intersections[path_index].tangent = tangent;
        }
    }
}

__device__ glm::vec3 fresnelDielectricEval(float etaI, float etaT, float cosThetaI) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    bool entering = cosThetaI > 0.f;
    if (!entering) {
        //swap etaI and etaT
        float temp = etaI;
        etaI = etaT;
        etaT = temp;

        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrt(max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2.f);
}

__device__ glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) {
    float cos_theta = std::fmin(dot(-uv, n), 1.f);
    glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    glm::vec3 r_out_parallel = -glm::sqrt(glm::abs(1.f - glm::length2(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

__device__ float cosTheta(glm::vec3 v1, glm::vec3 v2) {
    return glm::cos(glm::acos(glm::dot(v1, v2)));
}

__device__ float random(glm::vec2 st) {
    return glm::fract(glm::sin(glm::dot(glm::vec2(st.x, st.y),
        glm::vec2(12.9898, 78.233))) *
        43758.5453123);
}

__device__ float noise(glm::vec2 st) {
    glm::vec2 i = glm::floor(st);
    glm::vec2 f = glm::fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + glm::vec2(1.0, 0.0));
    float c = random(i + glm::vec2(0.0, 1.0));
    float d = random(i + glm::vec2(1.0, 1.0));

    glm::vec2 u = f * f * (3.f - 2.f * f);

    return glm::mix(a, b, u.x) +
        (c - a) * u.y * (1.0 - u.x) +
        (d - b) * u.x * u.y;
}

#define OCTAVES 6
__device__ float fbm(glm::vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitud = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitud * noise(st);
        st *= 2.;
        amplitud *= .5;
    }
    return value;
}

__global__ void shadeMaterials(int iter,
                               int num_paths,
                               int depth,
                               ShadeableIntersection* shadeableIntersections,
                               PathSegment* pathSegments,
                               Material* materials,
                               glm::vec4* texture_data,
                               glm::vec4* bumpmap_data,
                               int* tex_starts,
                               int* bump_starts,
                               glm::vec2* tex_dims,
                               glm::vec2* bump_dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > num_paths || pathSegments[idx].remainingBounces <= 0) {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) // if the intersection exists...
    {
        // Set up the RNG
        // LOOK: this is how you use thrust's RNG! Please look at
        // makeSeededRandomEngine as well.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor;
        if (material.isTexture) {
            int start_idx = tex_starts[material.tex_index];
            glm::vec2 dims = tex_dims[material.tex_index];

            glm::vec2 uv = intersection.uv;
            //pass width and height in, mult u * width, v * height
            int tex_x_idx = glm::fract(uv.x) * dims.x; //scale from 0..1 to 0..63
            int tex_y_idx = glm::fract(1.0f - uv.y) * dims.y; //scale from 0..1 to 0..63 
            int tex_1d_idx = start_idx + tex_y_idx * dims.x + tex_x_idx;

#define USETEXTURE 1
#if USETEXTURE
            materialColor = glm::vec3(texture_data[tex_1d_idx]);
#else
            //https://thebookofshaders.com/edit.php?log=161127201157
            float v0 = glm::mix(-1.0, 1.0, sin(uv.x * 14.0 + fbm(glm::vec2(uv.x, uv.x) * glm::vec2(100.0, 12.0)) * 8.0));
            float v1 = random(uv);
            float v2 = noise(uv * glm::vec2(200.0, 14.0)) - noise(uv * glm::vec2(1000.0, 64.0));

            glm::vec3 col = glm::vec3(0.860, 0.806, 0.574);
            col = glm::mix(col, glm::vec3(0.390, 0.265, 0.192), v0);
            col = glm::mix(col, glm::vec3(0.930, 0.493, 0.502), v1 * 0.5);
            col -= v2 * 0.2;
            materialColor = col;
#endif
        }
        else {
            materialColor = material.color;
        }

#define USEBUMPMAP 1
#if USEBUMPMAP
        if (material.isBumpmap) {
            int start_idx = bump_starts[material.bumpmap_index];
            glm::vec2 dims = bump_dims[material.bumpmap_index];

            glm::vec2 uv = intersection.uv;
            //pass width and height in, mult u * width, v * height
            int tex_x_idx = glm::fract(uv.x) * dims.x; //scale from 0..1 to 0..63
            int tex_y_idx = glm::fract(1.0f - uv.y) * dims.y; //scale from 0..1 to 0..63 
            int tex_1d_idx = start_idx + tex_y_idx * dims.x + tex_x_idx;

            glm::vec3& tangent = intersection.tangent;
            glm::vec3 bitangent = glm::cross(intersection.surfaceNormal, tangent);
            glm::mat3 nor_transform{glm::normalize(tangent), glm::normalize(bitangent), glm::normalize(intersection.surfaceNormal)};
            intersection.surfaceNormal = glm::normalize(nor_transform * glm::vec3(bumpmap_data[tex_1d_idx]));
        }
#endif

        PathSegment& curr_seg = pathSegments[idx];
        Ray& curr_ray = curr_seg.ray;

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0f) {
            curr_seg.color *= (materialColor * material.emittance);
            curr_seg.remainingBounces = 0;
        } 
        else  if (material.specular_transmissive.isSpecular == false) {
            //perfectly diffuse for now
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            glm::vec3 wi;
            scatterRay(curr_seg, isect_pt, intersection.surfaceNormal, material, rng, wi);

            wi = glm::normalize(wi);

            float costheta = cosTheta(wi, intersection.surfaceNormal);
            float pdf = costheta * INV_PI;
            if (pdf == 0.f) {
                curr_seg.remainingBounces = 0;
                return;
            }

            glm::vec3 bsdf = materialColor * INV_PI;
            float lambert = glm::abs(glm::dot(wi, intersection.surfaceNormal));

            curr_seg.color *= (bsdf * lambert) / pdf;

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
        }
        else if (material.specular_transmissive.isSpecular == true && material.specular_transmissive.isTransmissive == false) {
            //perfectly specular
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            glm::vec3 wi = glm::reflect(curr_ray.direction, intersection.surfaceNormal);

            wi = glm::normalize(wi);

            //took out lambert and INV_PI from bsdf
            glm::vec3 bsdf = materialColor;
            float lambert = glm::abs(glm::dot(wi, intersection.surfaceNormal));

            curr_seg.color *= (bsdf); //pdf = 1

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
        }
        else if (material.specular_transmissive.isSpecular == true && material.specular_transmissive.isTransmissive == true) {
            
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            float rand_num = u01(rng);

            glm::vec3 wi, bsdf;

            float etaA = material.specular_transmissive.eta.x;
            float etaB = material.specular_transmissive.eta.y;

            float costheta = cosTheta(curr_ray.direction, intersection.surfaceNormal);
            bool entering = intersection.outside;
            float etaI = entering ? etaA : etaB;
            float etaT = entering ? etaB : etaA;
            float eta = etaI / etaT;

            bool reflected = false;

            wi = refract(curr_ray.direction, intersection.surfaceNormal, etaI / etaT);

            float cosThetaI = dot(intersection.surfaceNormal, wi);
            float sin2ThetaI = max(0.f, 1.f - cosThetaI * cosThetaI);
            float sin2ThetaT = eta * eta * sin2ThetaI;

            if (rand_num < 0.5f || sin2ThetaI >= 1.f) {
                //using specular reflection
                reflected = true;
                wi = glm::reflect(curr_ray.direction, intersection.surfaceNormal);
                bsdf = materialColor;
            }
            else {

                //using specular refraction
                glm::vec3 T = materialColor / glm::abs(cosTheta(wi, intersection.surfaceNormal));
                bsdf = (glm::vec3(1.) - fresnelDielectricEval(etaI, etaT, glm::dot(nor, normalize(wi)))) * T;

                bsdf *= glm::abs(glm::dot(wi, intersection.surfaceNormal));
                
            }

            curr_seg.color *= (bsdf); //pdf = 1

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin;
            if (reflected) {
                new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            }
            else {
                new_origin = isect_pt - intersection.surfaceNormal * 0.01f;
            }
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
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

struct ShouldTerminate {
    __host__ __device__ bool operator()(const PathSegment& x)
    {
        return x.remainingBounces > 0;
    }
};

struct CompareMaterials
{
    __host__ __device__ bool operator()(const ShadeableIntersection& first, const ShadeableIntersection& second)
    {
        return first.materialId < second.materialId;
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
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
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
    int num_paths = dev_path_end - dev_paths; //just the pixel count for now

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
            dev_intersections,
            dev_triangles,
            hst_scene->triangle_count,
            dev_bvhnodes
        );
        checkCUDAError("compute intersections");
        cudaDeviceSynchronize();
        depth++;

#define SORTBYMATERIAL 1
#if SORTBYMATERIAL
        thrust::device_ptr<ShadeableIntersection> dev_inters_to_sort(dev_intersections);
        thrust::device_ptr<PathSegment> dev_paths_to_sort(dev_paths); //values
        thrust::stable_sort_by_key(dev_inters_to_sort, dev_inters_to_sort + num_paths, dev_paths_to_sort, CompareMaterials());
#endif

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_tex_data,
            dev_bumpmap_data,
            dev_tex_starts,
            dev_bump_starts,
            dev_tex_dims,
            dev_bump_dims
            );

#define USE_COMPACTION 1
#if USE_COMPACTION
        thrust::device_ptr<PathSegment> dev_paths_to_compact(dev_paths);
        thrust::device_ptr<PathSegment> last_elt = thrust::stable_partition(thrust::device, dev_paths_to_compact, dev_paths_to_compact + num_paths, ShouldTerminate());
        num_paths = last_elt.get() - dev_paths;
#endif

        iterationComplete = (depth >= traceDepth || num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //NOTE: changed N to pixelcount from num paths, still want to check paths that will be terminated
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
