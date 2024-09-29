#pragma once
#include "distribution1D.h"

class Distribution2D {
public:

    std::vector<Distribution1D> pConditionalV;
    Distribution1D pMarginal;

    // Distribution2D Public Methods
    Distribution2D(const float* data, int nu, int nv);
    Distribution2D() = default;;
    glm::vec2 SampleContinuous(const glm::vec2& u, float& pdf) const 
    {
        float pdfs[2];
        float d1 = pMarginal.sampleContinuous(u[1], pdfs[1]);
        float d0 = pConditionalV[static_cast<int>(d1)].sampleContinuous(u[0], pdfs[0]);
        pdf = pdfs[0] * pdfs[1];
        return glm::vec2(d0, d1);
    }

    float Pdf(const glm::vec2& p) const 
    {
        int iu = glm::clamp(int(p[0] * pConditionalV[0].Count()), 0,
            pConditionalV[0].Count() - 1);
        int iv =
            glm::clamp(int(p[1] * pMarginal.Count()), 0, pMarginal.Count() - 1);
        return pConditionalV[iv].func[iu] / pMarginal.funcInt;
    }
};

class DevDistribution2D
{
public:
	DevDistribution1D* pConditionalV = nullptr;
	DevDistribution1D pMarginal;

    void create(Distribution2D& srcSampler);

	void destroy();

	__host__ __device__ glm::vec2 sampleContinuous(const glm::vec2& u, float& pdf)const
	{
        float pdfs[2];
        float d1 = pMarginal.sampleContinuous(u[1], pdfs[1]);
        float d0 = pConditionalV[static_cast<int>(d1)].sampleContinuous(u[0], pdfs[0]);
        pdf = pdfs[0] * pdfs[1];
        return glm::vec2(d0, d1);
	}

    __host__ __device__ float getPdf(const glm::vec2& p)const
    {
        int iu = glm::clamp(int(p[0] * pConditionalV[0].Count()), 0,
            pConditionalV[0].Count() - 1);
        int iv =
            glm::clamp(int(p[1] * pMarginal.Count()), 0, pMarginal.Count() - 1);
        return pConditionalV[iv].func[iu] / pMarginal.funcInt;
    }
};