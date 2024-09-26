#pragma once
#include "cudaUtilities.h"

__global__ void updateTriangleTransform(Geom& geom, Triangle* triangles, int numTriangles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numTriangles) return;

	Triangle& tri = triangles[index];
#pragma unroll
	for (int j = 0; j < 3; j++)
	{
		tri.vertices[j] = glm::vec3(geom.transform * glm::vec4(tri.vertices[j], 1.0f));
		tri.normals[j] = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(tri.normals[j], 0.0f)));
	}
}

void updateTrianglesTransform(Geom& dev_geom, Triangle* dev_triangles)
{
	int numTriangles = dev_geom.triangleEndIdx - dev_geom.triangleStartIdx;
	int blockSize = 128;
	int numBlocks = (numTriangles + blockSize - 1) / blockSize;
	updateTriangleTransform << <numBlocks, blockSize >> > (dev_geom, dev_triangles, numTriangles);
}
