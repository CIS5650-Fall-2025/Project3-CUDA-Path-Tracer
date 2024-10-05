#include "distribution2D.h"

std::vector<DevDistribution1D> pConditionalVtemp;

Distribution2D::Distribution2D(const float* func, int width, int height) 
{
    pConditionalV.reserve(height);
    for (int v = 0; v < height; ++v) {
        // Compute conditional sampling distribution for $\tilde{v}$
        pConditionalV.emplace_back(&func[v * width], width);
    }
    // Compute marginal sampling distribution $p[\tilde{v}]$
    std::vector<float> marginalFunc;
    marginalFunc.reserve(height);
    for (int v = 0; v < height; ++v)
        marginalFunc.push_back(pConditionalV[v].funcInt);
    pMarginal = std::move(Distribution1D(marginalFunc));
}

Distribution2D::Distribution2D(const std::vector<float> &func, int width, int height) 
{
    pConditionalV.reserve(height);
    for (int v = 0; v < height; ++v) {
        // Compute conditional sampling distribution for $\tilde{v}$
        pConditionalV.emplace_back(&func[v * width], width);
    }
    // Compute marginal sampling distribution $p[\tilde{v}]$
    std::vector<float> marginalFunc;
    marginalFunc.reserve(height);
    for (int v = 0; v < height; ++v)
        marginalFunc.push_back(pConditionalV[v].funcInt);
    pMarginal = std::move(Distribution1D(marginalFunc));
}

void DevDistribution2D::create(Distribution2D& srcSampler)
{
	pMarginal.create(srcSampler.pMarginal);
	int size = srcSampler.pConditionalV.size();
	pConditionalVtemp.clear();
	pConditionalVtemp.resize(size);
	for (int i = 0; i < size; ++i)
	{
		pConditionalVtemp[i].create(srcSampler.pConditionalV[i]);
	}
	cudaMalloc(&pConditionalV, size * sizeof(DevDistribution1D));
	cudaMemcpy(pConditionalV, pConditionalVtemp.data(), size * sizeof(DevDistribution1D), cudaMemcpyHostToDevice);
	checkCUDAError("DevDistribution2D create()::pConditionalV");
}

void DevDistribution2D::destroy()
{
	int size = pMarginal.Count();
    pMarginal.destroy();
    if (!pConditionalVtemp.empty())
    {
        for (int i = pConditionalVtemp.size() - 1; i >= 0; --i)
        {
			pConditionalVtemp[i].destroy();
			pConditionalVtemp.pop_back();
        }
    }
    cudaSafeFree(pConditionalV);
}