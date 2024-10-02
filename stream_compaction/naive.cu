#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int* dev_odata;
        int* dev_idata;
        int* dev_tempdata;

        const int blockSize = 128;

        // TODO: __global__
        __global__ void scanInGPU(int n, int logVal, int* odata, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n)return;

            int offset = 1 << logVal;

            if (index - offset < 0) {
                odata[index] = idata[index];
            }
            else {
                odata[index] = idata[index - offset] + idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* tempdata = (int*)malloc(n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_tempdata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_tempdata failed!");

            cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_tempdata, tempdata, sizeof(int) * n, cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            for (int i = 1; i < ilog2ceil(n) + 1; i++) {
                scanInGPU << <fullBlocksPerGrid, blockSize >> > (n, i - 1, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }

            std::swap(dev_odata, dev_idata);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_tempdata);

            free(tempdata);
        }
    }
}
