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

        // TODO: __global__
        __global__ void kernelNaiveScan(int n, int offset, int* idata, int* odata) 
        {
            // Using the Naive algorithm from GPU Gems 3, Section 39.2.1.
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int* bufferA, * bufferB;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            cudaMalloc((void**)&bufferA, n * sizeof(int));
            checkCUDAError("cudaMalloc bufferA failed!");
            cudaMalloc((void**)&bufferB, n * sizeof(int));
            checkCUDAError("cudaMalloc bufferB failed!");

            // Copy from host input data to device input data
            cudaMemcpy(bufferA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // DONE

            // Calculate number of levels
            int numLevels = ilog2ceil(n);
            int offset;

            for(int d = 1; d <= numLevels; d++) {
                
                offset = 1 << (d - 1);

                kernelNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, offset, bufferA, bufferB);
                checkCUDAError("kernelNaiveScan failed!");
                
                // Sync threads before swapping
                cudaDeviceSynchronize();
                std::swap(bufferA, bufferB);
            }

            timer().endGpuTimer();

            // Insert identity and shift right to make exclusive
            // Copy result from device back to host
            odata[0] = 0;
            cudaMemcpy(odata + 1, bufferA, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from bufferA to odata failed!");

            // Free bufferA and bufferB memory
            cudaFree(bufferA);
            cudaFree(bufferB);
        }
    }
}