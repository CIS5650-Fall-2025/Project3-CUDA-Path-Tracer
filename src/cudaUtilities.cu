#pragma once
#include "cudaUtilities.h"

Triangle* dev_triangles = NULL;
Geom * dev_geoms = NULL;
int dev_numGeoms = 2;
Material* dev_materials = NULL;
int* dev_triTransforms = NULL;

void checkCUDAError(const char* msg)
{
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void initSceneCuda(Geom* geoms, Material* materials, Triangle* triangles, int numGeoms, int numMaterials, int numTriangles)
{
	cudaMalloc(&dev_geoms, numGeoms * sizeof(Geom));
	cudaMalloc(&dev_materials, numMaterials * sizeof(Material));
	cudaMalloc(&dev_triangles, numTriangles * sizeof(Triangle));
	cudaMalloc(&dev_triTransforms, numTriangles * sizeof(int));
	checkCUDAError("initSceneCuda");

	dev_numGeoms = numGeoms;
	cudaMemcpy(dev_geoms, geoms, numGeoms * sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_materials, materials, numMaterials * sizeof(Material), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_triangles, triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
	int blockSize = 128;
	int numBlocks = (numGeoms + blockSize - 1) / blockSize;
	updateTriangleTransformIndex << <numBlocks, blockSize >> > (dev_geoms, dev_triTransforms, numGeoms);

	numBlocks = (numTriangles + blockSize - 1) / blockSize;
	updateTriangleTransform << <numBlocks, blockSize >> > (dev_geoms, dev_triangles, dev_triTransforms, numGeoms, numTriangles);

	cudaMemcpy(triangles, dev_triangles, numTriangles * sizeof(Triangle), cudaMemcpyDeviceToHost);
	checkCUDAError("initSceneCuda");
	
}

__global__ void updateTriangleTransformIndex(Geom* dev_geoms, int* dev_triTransforms, int numGeoms)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numGeoms) return;

	for (int i = dev_geoms[idx].triangleStartIdx; i < dev_geoms[idx].triangleEndIdx; i++)
	{
		dev_triTransforms[i] = idx;
	}
}

__global__ void updateTriangleTransform(Geom* dev_geoms, Triangle* dev_triangles, int* dev_triTransforms, int numGeoms, int numTriangles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numTriangles) return;
	glm::mat4& transform = dev_geoms[dev_triTransforms[idx]].transform;
	
	// transform vertices
	dev_triangles[idx].vertices[0] = glm::vec3(transform * glm::vec4(dev_triangles[idx].vertices[0], 1.f));
	dev_triangles[idx].vertices[1] = glm::vec3(transform * glm::vec4(dev_triangles[idx].vertices[1], 1.f));
	dev_triangles[idx].vertices[2] = glm::vec3(transform * glm::vec4(dev_triangles[idx].vertices[2], 1.f));

	// transform normals
	dev_triangles[idx].normals[0] = glm::normalize(glm::vec3(glm::transpose(glm::inverse(transform)) * glm::vec4(dev_triangles[idx].normals[0], 0.f)));
	dev_triangles[idx].normals[1] = glm::normalize(glm::vec3(glm::transpose(glm::inverse(transform)) * glm::vec4(dev_triangles[idx].normals[1], 0.f)));
	dev_triangles[idx].normals[2] = glm::normalize(glm::vec3(glm::transpose(glm::inverse(transform)) * glm::vec4(dev_triangles[idx].normals[2], 0.f)));
	
}


void freeSceneCuda()
{
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_triangles);
}

void printGeoms()
{
	Geom* host_geoms = new Geom[dev_numGeoms];
	cudaMemcpy(host_geoms, dev_geoms, dev_numGeoms * sizeof(Geom), cudaMemcpyDeviceToHost);
	for (int i = 0; i < dev_numGeoms; ++i)
	{
		printf("geom type: %d\n", host_geoms[i].type);
	}
}