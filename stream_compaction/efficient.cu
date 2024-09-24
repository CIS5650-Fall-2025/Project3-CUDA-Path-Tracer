#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //variable initialization
        int* dev_sdata;
        int* dev_bdata;
        int* dev_odata;
        int* dev_idata;
        int* dev_origin;
        const int blockSize = 128;

        //GPU scan
        __global__ void upSweep(int n, int logVal, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n)return;

            int offset1 = 1 << (logVal + 1);
            int offset2 = 1 << logVal;
            if (index % offset1 == 0) {
                idata[index + offset1 - 1] += idata[index + offset2 - 1];
            }
        }

        __global__ void downSweep(int n, int logVal, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n)return;

            int offset1 = 1 << (logVal + 1);
            int offset2 = 1 << logVal;

            if (index % offset1 == 0) {
                int t = idata[index + offset1 - 1];
                idata[index + offset1 - 1] += idata[index + offset2 - 1];
                idata[index + offset2 - 1] = t;
            }
        }

        __global__ void shiftArray(int n, int* origin, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n)return;

            idata[index] += origin[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemcpy(dev_idata, idata, size * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            cudaMalloc((void**)&dev_origin, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_origin, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemcpy(dev_origin, idata, size * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            for (int i = 0; i <= ilog2ceil(n) - 1; i++) {
                upSweep << <fullBlocksPerGrid, blockSize >> > (size, i, dev_idata);
                checkCUDAError("cudaFunc upSweep failed!");
            }

            cudaMemset(dev_idata + size - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_idata + size - 1 failed!");

            for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
                downSweep << <fullBlocksPerGrid, blockSize >> > (size, i, dev_idata);
                checkCUDAError("cudaFunc downSweep failed!");
            }

            timer().endGpuTimer();
            shiftArray << <fullBlocksPerGrid, blockSize >> > (size, dev_origin, dev_idata);
            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_origin);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int size = pow(2, ilog2ceil(n));
            int* test = (int*)malloc(sizeof(int) * size);

            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemcpy(dev_idata, idata, size * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            cudaMalloc((void**)&dev_odata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemset(dev_odata, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_odata failed!");
            cudaMalloc((void**)&dev_bdata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_bdata failed!");
            cudaMemset(dev_bdata, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_bdata failed!");
            cudaMalloc((void**)&dev_sdata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_sdata failed!");
            cudaMemset(dev_sdata, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_sdata failed!");

            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bdata, dev_idata);

            cudaMemcpy(dev_sdata, dev_bdata, size * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int i = 0; i <= ilog2ceil(n) - 1; i++) {
                upSweep << <fullBlocksPerGrid, blockSize >> > (size, i, dev_sdata);
                checkCUDAError("cudaFunc upSweep failed!");
            }

            cudaMemset(dev_sdata + size - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_idata + size - 1 failed!");

            for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
                downSweep << <fullBlocksPerGrid, blockSize >> > (size, i, dev_sdata);
                checkCUDAError("cudaFunc downSweep failed!");
            }
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (size, dev_odata, dev_idata, dev_bdata, dev_sdata);

            timer().endGpuTimer();
            int num1 = 0;
            int num2 = 0;
            cudaMemcpy(&num1, dev_sdata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num2, dev_bdata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, (num1 + num2 ) * sizeof(int), cudaMemcpyDeviceToHost);
            
            //test
            //for (int i = 0; i < n; i++) {
            //    std::cout << idata[i] << " ";
            //}
            //std::cout << std::endl;
            //cudaMemcpy(test, dev_sdata, sizeof(int) * size, cudaMemcpyDeviceToHost);
            //for (int i = 0; i < size; i++) {
            //    std::cout << test[i] << " ";
            //}
            //std::cout << std::endl;
            //cudaMemcpy(test, dev_bdata, sizeof(int) * size, cudaMemcpyDeviceToHost);
            //for (int i = 0; i < size; i++) {
            //    std::cout << test[i] << " ";
            //}
            //std::cout << std::endl;

            //free
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bdata);
            cudaFree(dev_sdata);
            free(test);
            return num1 + num2;
        }
    }
}
