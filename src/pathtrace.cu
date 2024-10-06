#include <OpenImageDenoise/oidn.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <cmath>
#include <cstdio>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "interactions.h"
#include "intersections.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#define ERRORCHECK 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif  // _WIN32
  exit(EXIT_FAILURE);
#endif  // ERRORCHECK
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

__host__ __device__ glm::vec2 SampleUniformDiskConcentric(glm::vec2 u) {
  glm::vec2 uOffset = 2.0f * u - glm::vec2(1);
  if (uOffset.x == 0 && uOffset.y == 0) {
    return glm::vec2(0);
  }

  float theta, r;
  if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
    r = uOffset.x;
    theta = PI_OVER_FOUR * (uOffset.y / uOffset.x);
  } else {
    r = uOffset.y;
    theta = PI_OVER_TWO - PI_OVER_FOUR * (uOffset.x / uOffset.y);
  }
  return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, bool shift, bool getAvg,
                               glm::vec3* devScaledBuffer) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    glm::vec3 pix = image[index];
    if (shift) {
      pix = (pix + glm::vec3(1.0f)) * 0.5f;
    }

    glm::vec3 total;
    if (getAvg) {
      total = pix;
      pix /= (float)iter;
    } else {
      total = pix * (float)iter;
    }
    devScaledBuffer[index].x = total.x;
    devScaledBuffer[index].y = total.y;
    devScaledBuffer[index].z = total.z;

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
static glm::vec3* dev_normals = NULL;
static glm::vec3* dev_normals_total = NULL;
static glm::vec3* dev_albedos = NULL;
static glm::vec3* dev_albedos_total = NULL;
static glm::vec3* dev_image_denoised = NULL;
static glm::vec3* dev_normals_denoised = NULL;
static glm::vec3* dev_albedos_denoised = NULL;
static glm::vec3* dev_scaled_buffer = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static Triangle* dev_triangles = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

static thrust::device_ptr<PathSegment> thrust_dev_paths = NULL;
static thrust::device_ptr<ShadeableIntersection> thrust_dev_intersections = NULL;

static OIDNDevice oidnDevice;
static OIDNFilter oidnFilter;
static OIDNFilter oidnNormalFilter;
static OIDNFilter oidnAlbedoFilter;

static OIDNBuffer oidnColorBuffer;
static OIDNBuffer oidnNormalBuffer;
static OIDNBuffer oidnAlbedoBuffer;

static OIDNBuffer oidnOutputBuffer;
static OIDNBuffer oidnOutputNormalBuffer;
static OIDNBuffer oidnOutputAlbedoBuffer;

void InitDataContainer(GuiDataContainer* imGuiData) { guiData = imGuiData; }

void initOIDN() {
  int deviceId = 0;
  cudaStream_t stream;
  oidnDevice = oidnNewCUDADevice(&deviceId, &stream, 1);
  oidnCommitDevice(oidnDevice);

  const Camera& cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;
  const int size = pixelcount * sizeof(glm::vec3);
  oidnColorBuffer = oidnNewSharedBuffer(oidnDevice, dev_image, size);
  oidnNormalBuffer = oidnNewSharedBuffer(oidnDevice, dev_normals, size);
  oidnAlbedoBuffer = oidnNewSharedBuffer(oidnDevice, dev_albedos, size);
  oidnOutputBuffer = oidnNewSharedBuffer(oidnDevice, dev_image_denoised, size);
  oidnOutputNormalBuffer = oidnNewSharedBuffer(oidnDevice, dev_normals_denoised, size);
  oidnOutputAlbedoBuffer = oidnNewSharedBuffer(oidnDevice, dev_albedos_denoised, size);

  oidnFilter = oidnNewFilter(oidnDevice, "RT");
  oidnSetFilterImage(oidnFilter, "color", oidnColorBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0, 0,
                     0);
  oidnSetFilterImage(oidnFilter, "normal", oidnOutputNormalBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x,
                     cam.resolution.y, 0, 0, 0);
#if ERRORCHECK
  const char* errorMiddle;
  if (oidnGetDeviceError(oidnDevice, &errorMiddle) != OIDN_ERROR_NONE) {
    printf("OIDN Error: %s\n", errorMiddle);
    exit(1);
  }
#endif
  oidnSetFilterImage(oidnFilter, "albedo", oidnOutputAlbedoBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x,
                     cam.resolution.y, 0, 0, 0);
  oidnSetFilterImage(oidnFilter, "output", oidnOutputBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0,
                     0, 0);
  oidnSetFilterBool(oidnFilter, "hdr", true);
  oidnCommitFilter(oidnFilter);

  oidnNormalFilter = oidnNewFilter(oidnDevice, "RT");
  oidnSetFilterImage(oidnNormalFilter, "normal", oidnNormalBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x,
                     cam.resolution.y, 0, 0, 0);
  oidnSetFilterImage(oidnNormalFilter, "output", oidnOutputNormalBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x,
                     cam.resolution.y, 0, 0, 0);
  oidnCommitFilter(oidnNormalFilter);

  oidnAlbedoFilter = oidnNewFilter(oidnDevice, "RT");
  oidnSetFilterImage(oidnAlbedoFilter, "albedo", oidnAlbedoBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x,
                     cam.resolution.y, 0, 0, 0);
  oidnSetFilterImage(oidnAlbedoFilter, "output", oidnOutputAlbedoBuffer, OIDN_FORMAT_FLOAT3, cam.resolution.x,
                     cam.resolution.y, 0, 0, 0);
  oidnCommitFilter(oidnAlbedoFilter);

#if ERRORCHECK
  const char* error;
  if (oidnGetDeviceError(oidnDevice, &error) != OIDN_ERROR_NONE) {
    printf("OIDN Error: %s\n", error);
    exit(1);
  }
#endif
}

void pathtraceInit(Scene* scene) {
  hst_scene = scene;

  const Camera& cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_normals, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_normals, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_normals_total, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_normals_total, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_albedos, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_albedos, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_albedos_total, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_albedos_total, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_image_denoised, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image_denoised, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_normals_denoised, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_normals_denoised, 0, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_albedos_denoised, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_albedos_denoised, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
  thrust_dev_paths = thrust::device_pointer_cast(dev_paths);

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
  cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
  thrust_dev_intersections = thrust::device_pointer_cast(dev_intersections);

  cudaMalloc(&dev_scaled_buffer, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_scaled_buffer, 0, pixelcount * sizeof(glm::vec3));

  initOIDN();

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  oidnReleaseBuffer(oidnColorBuffer);
  oidnReleaseBuffer(oidnNormalBuffer);
  oidnReleaseBuffer(oidnAlbedoBuffer);
  oidnReleaseBuffer(oidnOutputBuffer);
  oidnReleaseBuffer(oidnOutputNormalBuffer);
  oidnReleaseBuffer(oidnOutputAlbedoBuffer);
  oidnReleaseFilter(oidnFilter);
  oidnReleaseFilter(oidnNormalFilter);
  oidnReleaseFilter(oidnAlbedoFilter);
  oidnReleaseDevice(oidnDevice);

  cudaFree(dev_image);  // no-op if dev_image is null
  cudaFree(dev_normals);
  cudaFree(dev_normals_total);
  cudaFree(dev_albedos);
  cudaFree(dev_albedos_total);
  cudaFree(dev_image_denoised);
  cudaFree(dev_normals_denoised);
  cudaFree(dev_albedos_denoised);
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  cudaFree(dev_triangles);
  cudaFree(dev_intersections);
  cudaFree(dev_scaled_buffer);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments,
                                      bool antiAliasing) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x >= cam.resolution.x || y >= cam.resolution.y) {
    return;
  }

  int index = x + (y * cam.resolution.x);
  PathSegment& segment = pathSegments[index];

  segment.ray.origin = cam.position;
  segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

  float xOffset = 0.0f;
  float yOffset = 0.0f;
  thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
  if (antiAliasing) {
    thrust::uniform_real_distribution<float> u(-0.5, 0.5);
    xOffset = u(rng);
    yOffset = u(rng);
  }
  glm::vec3 originalDirection =
      glm::normalize(cam.view - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + xOffset) -
                     cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + yOffset));

  if (cam.lensRadius > 0) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 lensPoint = cam.lensRadius * SampleUniformDiskConcentric(glm::vec2(u01(rng), u01(rng)));

    float z = glm::dot(originalDirection, cam.view);
    float ft = cam.focalDistance / z;
    glm::vec3 focalPoint = cam.position + originalDirection * ft;

    segment.ray.origin = cam.position + (lensPoint.x * cam.right + lensPoint.y * cam.up);
    segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
  } else {
    segment.ray.direction = originalDirection;
  }

  segment.pixelIndex = index;
  segment.remainingBounces = traceDepth;
  segment.needsNormal = true;
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths, PathSegment* pathSegments, Geom* geoms,
                                     Triangle* triangles, int geoms_size, ShadeableIntersection* intersections,
                                     bool enableBVC) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths) {
    PathSegment pathSegment = pathSegments[path_index];
    if (pathSegment.remainingBounces <= 0) {
      return;
    }

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++) {
      Geom& geom = geoms[i];

      if (geom.type == CUBE) {
        t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
      } else if (geom.type == SPHERE) {
        t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
      } else if (geom.type == MESH) {
        t = meshIntersectionTest(geom, triangles, pathSegment.ray, enableBVC, tmp_intersect, tmp_normal, outside);
      }

      // Compute the minimum t from the intersection tests to determine what
      // scene geometry object was hit first.
      if (t > 0.0f && t_min > t) {
        t_min = t;
        hit_geom_index = i;
        intersect_point = tmp_intersect;
        normal = tmp_normal;
      }
    }

    if (hit_geom_index == -1) {
      intersections[path_index].t = -1.0f;
    } else {
      // The ray hits something
      intersections[path_index].t = t_min;
      intersections[path_index].materialId = geoms[hit_geom_index].materialid;
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
__global__ void shadeMaterial(int iter, int num_paths, ShadeableIntersection* shadeableIntersections,
                              PathSegment* pathSegments, Material* materials) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_paths) {
    return;
  }

  PathSegment pathSegment = pathSegments[idx];

  ShadeableIntersection intersection = shadeableIntersections[idx];
  if (intersection.t <= 0.0f) {
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    if (pathSegment.needsNormal) {
      pathSegment.needsNormal = false;
      pathSegment.normal = glm::vec3(0.0f);
      pathSegment.albedo = glm::vec3(0.0f);

      // Set to background color if missed only on the first bounce
      pathSegment.color = glm::vec3(0.0f);
    }
    pathSegment.remainingBounces = 0;
    pathSegments[idx] = pathSegment;
    return;
  }

  // Set up the RNG
  // LOOK: this is how you use thrust's RNG! Please look at
  // makeSeededRandomEngine as well.
  thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

  Material material = materials[intersection.materialId];
  glm::vec3 materialColor = material.color;

  if (material.emittance > 0.0f) {
    // If the material indicates that the object was a light, "light" the ray
    pathSegment.color *= (materialColor * material.emittance);
    pathSegment.remainingBounces = 0;
  } else {
    // Otherwise, do lighting computation
    glm::vec3 intersectionPoint = getPointOnRay(pathSegment.ray, intersection.t);
    scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rng);
    pathSegment.remainingBounces--;
  }

  if (pathSegment.needsNormal) {
    pathSegment.needsNormal = false;
    pathSegment.normal = intersection.surfaceNormal;
    pathSegment.albedo = material.color;
  }

  pathSegments[idx] = pathSegment;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int iter, int nPaths, glm::vec3* image, glm::vec3* normals, glm::vec3* normals_total,
                            glm::vec3* albedos, glm::vec3* albedos_total, PathSegment* iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;

    glm::vec3 new_normal_total = normals_total[iterationPath.pixelIndex] + iterationPath.normal;
    glm::vec3 new_albedo_total = albedos_total[iterationPath.pixelIndex] + iterationPath.albedo;

    normals_total[iterationPath.pixelIndex] = new_normal_total;
    albedos_total[iterationPath.pixelIndex] = new_albedo_total;

    normals[iterationPath.pixelIndex] = glm::normalize(new_normal_total / (float)(iter));
    albedos[iterationPath.pixelIndex] = new_albedo_total / (float)(iter);
  }
}

struct include_path_if {
  __host__ __device__ bool operator()(const PathSegment& path) { return path.remainingBounces > 0; }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera& cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  // 2D block for generating ray from camera
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d((cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
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

  generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, guiData->antiAliasing);
  checkCUDAError("generate camera ray");

  int depth = 0;
  PathSegment* dev_path_end = dev_paths + pixelcount;
  const int num_paths = dev_path_end - dev_paths;
  int num_paths_remaining = num_paths;

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  while (num_paths_remaining > 0 && depth < traceDepth) {
    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // tracing
    dim3 numblocksPathSegmentTracing = (num_paths_remaining + blockSize1d - 1) / blockSize1d;
    computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(depth, num_paths_remaining, dev_paths, dev_geoms,
                                                                       dev_triangles, hst_scene->geoms.size(),
                                                                       dev_intersections, guiData->enableBVC);
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
    depth++;

    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // Compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

    if (guiData->sortMaterials) {
      thrust::sort_by_key(thrust::device, thrust_dev_intersections, thrust_dev_intersections + num_paths_remaining,
                          thrust_dev_paths);
    }

    shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(iter, num_paths_remaining, dev_intersections, dev_paths,
                                                                dev_materials);

    thrust::device_ptr<PathSegment> new_end =
        thrust::partition(thrust::device, thrust_dev_paths, thrust_dev_paths + num_paths_remaining, include_path_if());
    num_paths_remaining = new_end - thrust_dev_paths;

    if (guiData != NULL) {
      guiData->TracedDepth = depth;
    }
  }

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  finalGather<<<numBlocksPixels, blockSize1d>>>(iter, num_paths, dev_image, dev_normals, dev_normals_total, dev_albedos,
                                                dev_albedos_total, dev_paths);

  ///////////////////////////////////////////////////////////////////////////

  glm::vec3* result;
  bool shift = guiData->showNormals;
  bool getAvg = !(guiData->showNormals || guiData->showAlbedos);
  bool useDenoised = (guiData->denoiseInterval > 0);
  if (useDenoised && (iter - 1) % guiData->denoiseInterval == 0) {
    oidnExecuteFilter(oidnNormalFilter);
    oidnExecuteFilter(oidnAlbedoFilter);
    oidnExecuteFilter(oidnFilter);
  }
  if (guiData->showNormals) {
    result = useDenoised ? dev_normals_denoised : dev_normals;
  } else if (guiData->showAlbedos) {
    result = useDenoised ? dev_albedos_denoised : dev_albedos;
  } else {
    result = useDenoised ? dev_image_denoised : dev_image;
  }

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, result, shift, getAvg, dev_scaled_buffer);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_scaled_buffer, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}
