#include "distribution2D.h"

Distribution2D::Distribution2D(const float* func, int nu, int nv) {
    pConditionalV.reserve(nv);
    for (int v = 0; v < nv; ++v) {
        // Compute conditional sampling distribution for $\tilde{v}$
        pConditionalV.emplace_back(&func[v * nu], nu);
    }
    // Compute marginal sampling distribution $p[\tilde{v}]$
    std::vector<float> marginalFunc;
    marginalFunc.reserve(nv);
    for (int v = 0; v < nv; ++v)
        marginalFunc.push_back(pConditionalV[v].funcInt);
    pMarginal = std::move(Distribution1D(marginalFunc));
}

void DevDistribution2D::create(Distribution2D& srcSampler)
{
	pMarginal.create(srcSampler.pMarginal);
	int size = srcSampler.pConditionalV.size();
	std::vector<DevDistribution1D> pConditionalVtemp(size);
	for (int i = 0; i < size; ++i)
	{
		pConditionalVtemp[i].create(srcSampler.pConditionalV[i]);
	}
	cudaMalloc(&pConditionalV, (size + 1) * sizeof(float));
	cudaMemcpy(pConditionalV, pConditionalVtemp.data(), size * sizeof(DevDistribution1D), cudaMemcpyHostToDevice);
	checkCUDAError("DevDistribution2D create()::pConditionalV");
}

void DevDistribution2D::destroy()
{
	int size = pMarginal.Count();
    pMarginal.destroy();
	for (int i = 0; i < size; ++i)
	{
		pConditionalV[i].destroy();
	}
}