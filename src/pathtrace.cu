#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) return;

	fprintf(stderr, "CUDA error");
	if (file) fprintf(stderr, " (%s:%d)", file, line);

	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
	getchar();
#endif // _WIN32
	exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}


__device__ glm::vec2 sampleRandomStratified(glm::vec2 uniform, int numSample, int numCells, bool enable) {
	if (enable) {
		// Samples (int) sqrt(numTotalSamples) from a stratified distribution
		// If more samples are sampled, they will be random over the entire domain
		const int numCellsPerSide = (int)sqrtf(numCells);
		const float gridLength = 1.0f / numCellsPerSide;

		if (numSample >= numCellsPerSide * numCellsPerSide) return uniform;

		glm::vec2 gridIdx;
		gridIdx.y = numSample / numCellsPerSide;
		gridIdx.x = numSample - gridIdx.y * numCellsPerSide;

		return (gridIdx + uniform) * gridLength;
	}
	else {
		return uniform;
	}
}

// taken from https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#sec:unit-disk-sample
__device__ glm::vec2 transformToDisk(const glm::vec2 squareInput) {
	glm::vec2 offsetInput = 2.0f * squareInput - glm::vec2{ 1.0f };
	if (offsetInput.x == 0.0f && offsetInput.y == 0.0f) return { 0, 0 };

	float r, theta;

	if (fabsf(offsetInput.x) > fabsf(offsetInput.y)) {
		r = offsetInput.x;
		theta = PI_OVER_FOUR * offsetInput.y / offsetInput.x;
	}
	else {
		r = offsetInput.y;
		theta = PI_OVER_TWO - PI_OVER_FOUR * offsetInput.x / offsetInput.y;
	}

	return r * glm::vec2{ cosf(theta), sinf(theta) };
}

// taken from https://en.wikipedia.org/wiki/SRGB#Transformation
__host__ __device__ inline float convertLinearToSRGB(float linear) {
	if (linear <= 0.0031308f)
		return linear * 12.92f;
	else
		return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		// Clamp color to valid RGB values and apply gamma correction
		glm::ivec3 color;
		color.x = glm::clamp((int)(convertLinearToSRGB(pix.x / iter) * 255.0), 0, 255);
		color.y = glm::clamp((int)(convertLinearToSRGB(pix.y / iter) * 255.0), 0, 255);
		color.z = glm::clamp((int)(convertLinearToSRGB(pix.z / iter) * 255.0), 0, 255);

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
static bool* dev_conditionBuffer = NULL;
static int* dev_keyBufferPaths = NULL;
static int* dev_keyBufferIntersections = NULL;
static Triangle* dev_triangles = NULL;
static cudaArray_t* cuArrays = NULL;
static cudaTextureObject_t* texObjs = NULL;
static cudaTextureObject_t* dev_texObjs = NULL;

void InitDataContainer(GuiDataContainer* imGuiData) {
	guiData = imGuiData;
}

void textureInit(Texture& texture, unsigned int index) {
	// Allocate CUDA array in device memory with 8 bit RGBA
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	cudaMallocArray(
		&cuArrays[index],
		&channelDesc,
		texture.width,
		texture.height
	);

	cudaMemcpy2DToArray(
		cuArrays[index],
		0,
		0,
		texture.data.data(),
		texture.width * 4 * sizeof(uint8_t),
		texture.width * 4 * sizeof(uint8_t),
		texture.height,
		cudaMemcpyHostToDevice
	);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArrays[index];

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat; // required for linear filtering
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaCreateTextureObject(&texObjs[index], &resDesc, &texDesc, NULL);

	cudaMemcpy(&dev_texObjs[index], &texObjs[index], sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
}

void pathtraceInit(Scene* scene) {
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

	if (scene->triangles.size() > 0) {
		cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
		cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	}

	cudaMalloc(&dev_conditionBuffer, pixelcount * sizeof(bool));
	cudaMalloc(&dev_keyBufferPaths, pixelcount * sizeof(int));
	cudaMalloc(&dev_keyBufferIntersections, pixelcount * sizeof(int));

	int numTextures = scene->textures.size();
	cuArrays = new cudaArray_t[numTextures];
	texObjs = new cudaTextureObject_t[numTextures];
	cudaMalloc(&dev_texObjs, numTextures * sizeof(cudaTextureObject_t));
	for (int i = 0; i < numTextures; ++i) {
		textureInit(scene->textures[i], i);
	}

	checkCUDAError("pathtraceInit");
}

void pathtraceFree(Scene* scene) {
	// no-op if pointer is null
	cudaFree(dev_image);
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_triangles);

	cudaFree(dev_conditionBuffer);
	cudaFree(dev_keyBufferPaths);
	cudaFree(dev_keyBufferIntersections);

	// Destroy texture objects
	if (cuArrays != nullptr) {
		for (int i = 0; i < scene->textures.size(); ++i) {
			cudaFreeArray(cuArrays[i]);
			cudaDestroyTextureObject(texObjs[i]);
		}
		delete[] cuArrays;
		delete[] texObjs;
	}

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
	PathSegment* pathSegments, bool enableAA, bool enableDOF, bool enableStratified, int numCells) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) return;

	int index = x + (y * cam.resolution.x);
	PathSegment& segment = pathSegments[index];
	Ray& ray = segment.ray;

	ray.origin = cam.position;
	segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

	if (enableAA) {
		thrust::default_random_engine rng_aa = makeSeededRandomEngine(iter, index, -1);
		thrust::uniform_real_distribution<float> aaOffset(0, 1);
		glm::vec2 aaOffsetVec = sampleRandomStratified(glm::vec2{ aaOffset(rng_aa), aaOffset(rng_aa) }, iter, numCells, enableStratified);
		ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f - aaOffsetVec.x)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f - aaOffsetVec.y)
		);
	}
	else {
		ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)(cam.resolution.x + 1) * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)(cam.resolution.y + 1) * 0.5f)
		);
	}

	if (enableDOF) {
		// use different random engines for DoF and AA
		thrust::default_random_engine rng_dof = makeSeededRandomEngine(iter, index, -2);
		thrust::uniform_real_distribution<float> dofUniform(0, 1);

		glm::vec2 aperturePoint = cam.aperture * transformToDisk(
			sampleRandomStratified(glm::vec2{ dofUniform(rng_dof), dofUniform(rng_dof) }, iter, numCells, enableStratified)
		);
		float perpendicularRayDirection = glm::dot(ray.direction, cam.view);
		float t = cam.focalDistance / perpendicularRayDirection;

		glm::vec3 focusPoint = ray.origin + t * ray.direction;
		ray.origin += aperturePoint.x * cam.right + aperturePoint.y * cam.up;
		ray.direction = glm::normalize(focusPoint - ray.origin);
	}

	segment.pixelIndex = index;
	segment.remainingBounces = traceDepth;
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
	Triangle* triangles,
	bool enableBBCheck)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths) return;

	PathSegment pathSegment = pathSegments[path_index];

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	glm::vec3 barycentricCoords;
	int triangleIdx;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;
	GeomType geomType;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	glm::vec3 tmp_barycentricCoords{ -1.0f };
	int tmp_triangleIdx{ -1 };

	// naive parse through global geoms
	for (int i = 0; i < geoms_size; i++) {
		Geom& geom = geoms[i];

		switch (geom.type) {
		case CUBE:
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			break;
		case SPHERE:
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			break;
		case MESH:
			t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal,
					tmp_barycentricCoords, tmp_triangleIdx, outside, triangles, enableBBCheck);
			break;
		}

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t < t_min) {
			t_min = t;
			hit_geom_index = i;
			intersect_point = tmp_intersect;
			normal = tmp_normal;
			barycentricCoords = tmp_barycentricCoords;
			triangleIdx = tmp_triangleIdx;
			geomType = geom.type;
		}
	}

	if (hit_geom_index == -1) {
		intersections[path_index].t = -1.0f;
	}
	else {
		// The ray hits something
		intersections[path_index].t = t_min;
		intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].geomType = geomType;
		if (triangleIdx != -1) {
			intersections[path_index].barycentricCoords = barycentricCoords;
			intersections[path_index].triangleIdx = triangleIdx;
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
	int numPaths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	Triangle* triangles,
	cudaTextureObject_t* texObjs,
	int depth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPaths) return;

	PathSegment& path = pathSegments[idx];
	if (path.remainingBounces <= 0) return;

	ShadeableIntersection intersection = shadeableIntersections[idx];
	// if the intersection exists... 
	if (intersection.t > 0.0f) {
		Material material = materials[intersection.materialId];
		glm::vec3 materialColor;
		if (intersection.geomType == MESH) {
			Triangle& triangle = triangles[shadeableIntersections[idx].triangleIdx];
			glm::vec2 uv = triangle.uv1 * intersection.barycentricCoords.x
				+ triangle.uv2 * intersection.barycentricCoords.y
				+ triangle.uv3 * intersection.barycentricCoords.z;

			//float4 colorRGBA = tex2D<float4>(texObjs[material.textureID], uv.x, uv.y);
			//float4 colorRGBA = tex2D<float4>(texObjs[0], uv.x, uv.y);
			float4 colorRGBA = tex2D<float4>(texObjs[0], uv.x, uv.y);
			materialColor = glm::vec3{ colorRGBA.x, colorRGBA.y, colorRGBA.z };
			//materialColor = glm::normalize(intersection.barycentricCoords);
		}
		else {
			materialColor = material.color;
		}

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 1.0f) {
			path.color *= (materialColor * material.emittance);
			// Assume that emittors do not reflect any light
			path.remainingBounces = 0;
		}
		else {
			// Set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
			// thrust::uniform_real_distribution<float> u01(0, 1);

			// Compute intersection point on surface
			glm::vec3 intersectionPoint = getPointOnRay(path.ray, intersection.t);

			if (!material.hasRefractive) path.color *= materialColor;
			// glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction)* materialColor;

			scatterRay(path, intersectionPoint, intersection.surfaceNormal, material, rng, iter);

			--path.remainingBounces;
		}
	}
	else {
		// If there was no intersection, color the ray black.
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int pixelCount, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= pixelCount) return;

	PathSegment iterationPath = iterationPaths[index];
	image[iterationPath.pixelIndex] += iterationPath.color;
}

__global__ void computeConditionBufferAndPartialImage(PathSegment* paths, int N, bool* conditionBuffer, glm::vec3* image) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	PathSegment iterationPath = paths[index];

	if (iterationPath.remainingBounces <= 0) {
		conditionBuffer[index] = true;
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
	else {
		conditionBuffer[index] = false;
	}
}

__global__ void computeKeyBuffers(const ShadeableIntersection* intersections, int N, int* keyBuffer1, int* keyBuffer2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	int materialId = intersections[index].materialId;

	keyBuffer1[index] = materialId;
	keyBuffer2[index] = materialId;
}

dim3 computeBlockCount1D(unsigned int N, unsigned int blockSize) {
	return dim3{ (N + blockSize - 1) / blockSize };
}

void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelCount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	if (guiData == nullptr) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (
			cam, iter, traceDepth, dev_paths, true, false, false, 225);
	}
	else {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (
			cam, iter, traceDepth, dev_paths, guiData->antiAliasing, guiData->depthOfField, guiData->stratified, guiData->stratNumCells);
	}

	checkCUDAError("generate camera ray");

	int depth = 0;
	int numPaths = pixelCount;
	PathSegment* dev_path_end{ dev_paths + numPaths };

	thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
	thrust::device_ptr<PathSegment> thrust_paths_end(dev_path_end);
	thrust::device_ptr<bool> thrust_conditionBuffer(dev_conditionBuffer);
	thrust::device_ptr<ShadeableIntersection> thrust_intersections(dev_intersections);
	thrust::device_ptr<int> thrust_keyBufferPaths(dev_keyBufferPaths);
	thrust::device_ptr<int> thrust_keyBufferIntersections(dev_keyBufferIntersections);

	dim3 numBlocks1d{ computeBlockCount1D(numPaths, blockSize1d) };

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, numPaths * sizeof(ShadeableIntersection));

		// tracing
		if (guiData == nullptr) {
			computeIntersections<<<numBlocks1d, blockSize1d>>>(
				depth,
				numPaths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_intersections,
				dev_triangles,
				true);
		}
		else {
		computeIntersections<<<numBlocks1d, blockSize1d>>>(
			depth,
			numPaths,
			dev_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_intersections,
			dev_triangles,
			guiData->bbCheck);
		}
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		if (guiData != nullptr && guiData->materialSort) {
			computeKeyBuffers<<<numBlocks1d, blockSize1d>>>(
				dev_intersections, numPaths, dev_keyBufferPaths, dev_keyBufferIntersections);
			checkCUDAError("computeKey");

			// Sorting twice is faster than using zip_iterator and uses less memory
			// than sorting an index map and using gather
			thrust::sort_by_key(
				thrust_keyBufferPaths, thrust_keyBufferPaths + numPaths, thrust_paths);
			checkCUDAError("sorting paths");

			thrust::sort_by_key(
				thrust_keyBufferIntersections, thrust_keyBufferIntersections + numPaths, thrust_intersections);
			checkCUDAError("sorting intersections");
		}

		shadeMaterial<<<numBlocks1d, blockSize1d>>>(
			iter,
			numPaths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_triangles,
			dev_texObjs,
			depth);
		checkCUDAError("shadeMaterial");
		cudaDeviceSynchronize();

		if (guiData == nullptr || guiData->streamCompaction) {
			computeConditionBufferAndPartialImage<<<numBlocks1d, blockSize1d>>>(
				dev_paths, numPaths, dev_conditionBuffer, dev_image);

			// Removes entries with 1s in the conditionBuffer
			thrust_paths_end = thrust::remove_if(
				thrust_paths, thrust_paths + numPaths, thrust_conditionBuffer, thrust::identity<bool>());
			cudaDeviceSynchronize();

			numPaths = thrust_paths_end - thrust_paths;
			numBlocks1d = computeBlockCount1D(numPaths, blockSize1d);

			// All rays have been terminated
			if (numPaths == 0) iterationComplete = true;
		}

		// Maximum depth reached
		if (depth == traceDepth) iterationComplete = true;

		if (guiData != nullptr) guiData->TracedDepth = depth;
	}

	if (guiData == nullptr || guiData->streamCompaction) {
		if (numPaths) {
			// Assemble the rest of this iteration and apply it to the image
			finalGather<<<numBlocks1d, blockSize1d>>>(numPaths, dev_image, dev_paths);
			checkCUDAError("finalGather");
		}
	}
	else {
		// Assemble this iteration and apply it to the image
		finalGather<<<numBlocks1d, blockSize1d>>>(pixelCount, dev_image, dev_paths);
		checkCUDAError("finalGather");
	}

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
	checkCUDAError("sendImageToPBO");

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}