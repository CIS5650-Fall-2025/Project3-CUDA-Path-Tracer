#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "cudaUtils.hpp"

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables#Distribution1D
// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#sec:sample-discrete-2d
class Distribution1D
{
public:
	std::vector<float> func, cdf;
	float funcInt;

	Distribution1D() = default;

	Distribution1D(std::vector<float> vals);

	Distribution1D(const float* vals, size_t n);

	size_t Count() const;

	float sampleContinuous(float u, float& pdf);

	size_t sampleDiscrete(float u, float& pdf);
};

class DevDistribution1D
{
public:
	float* func = nullptr, * cdf = nullptr;
	float funcInt = 0;
	size_t size = 0;

	void create(Distribution1D& hstSampler);

	void destroy();

	__host__ __device__ float sampleContinuous(float u, float& pdf)const
	{
		u = glm::clamp(u, 0.f, 1.f);
		size_t left = 0, right = size;
		while (right > left)
		{
			size_t mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		size_t offset = glm::clamp(right - 1, size_t(0), size - 1);

		pdf = func[offset] / funcInt;
		float du = u - cdf[offset];
		if ((cdf[offset + 1] - cdf[offset]) > 0)
		{
			du /= (cdf[offset + 1] - cdf[offset]);
		}
		else
		{
			du = 0;
		}
		return (offset + du) / size;
	}

	__host__ __device__ size_t sampleDiscrete(float u, float& pdf)const
	{
		u = glm::clamp(u, 0.f, 1.f);
		size_t left = 0, right = size;
		while (right > left)
		{
			size_t mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		volatile float t1 = cdf[right];
		size_t offset = glm::clamp(right - 1, size_t(0), size - 1);
		volatile float t2 = cdf[right - 1];

		pdf = func[offset] / funcInt;

		return offset;
	}

	__host__ __device__ float getPdf(size_t index)const
	{
		float pdf = func[index] / funcInt;

		return pdf;
	}
};