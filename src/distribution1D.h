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

	Distribution1D(const float* vals, int n);

	Distribution1D(const Distribution1D& other);

	Distribution1D& operator=(Distribution1D&& other) noexcept;

	int Count() const;

	float sampleContinuous(float u, float& pdf)const;

	int sampleDiscrete(float u, float& pdf)const;
};

class DevDistribution1D
{
public:
	float* func = nullptr, * cdf = nullptr;
	float funcInt = 0;
	int size = 0;

	void create(Distribution1D& srcSampler);

	void destroy();

	__host__ __device__ float sampleContinuous(float u, float& pdf)const
	{
		u = glm::clamp(u, 0.f, 1.f);
		int left = 0, right = size;
		while (right > left)
		{
			int mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		int offset = glm::clamp(right - 1, int(0), size - 1);

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

	__host__ __device__ int sampleDiscrete(float u, float& pdf)const
	{
		u = glm::clamp(u, 0.f, 1.f);
		int left = 0, right = size;
		while (right > left)
		{
			int mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		int offset = glm::clamp(right - 1, int(0), size - 1);

		pdf = func[offset] / funcInt;

		return offset;
	}

	__host__ __device__ float getPdf(int index)const
	{
		float pdf = func[index] / funcInt;

		return pdf;
	}

	__host__ __device__ int Count()const
	{
		return size;
	}
};