#include <cuda.h>
#include <cuda_runtime.h>
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

        // Helper functions
        // Up-Sweep - parallel reduction
        __global__ void upSweep(int n, int d, int* data)
        {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            int index = idx << (d + 1);

            if (index + (1 << (d + 1)) - 1 < n) 
            {
                // From the slides:
                // x[k + 2d+1 – 1] += x[k + 2d – 1];
                data[index + (1 << (d + 1)) - 1] += data[index + (1 << d) - 1];
            }
        }

        // Down-Sweep - traverse back down the tree using partial sums to build the scan in place.
        __global__ void downSweep(int n, int d, int* data)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int index = idx << (d + 1);

            if (index + (1 << (d + 1)) - 1 < n) 
            {
                // Save left child
                int temp = data[index + (1 << d) - 1];

                // Set left child to this node’s value
                data[index + (1 << d) - 1] = data[index + (1 << (d + 1)) - 1];

                // Set right child to old left value + this node’s value
                data[index + (1 << (d + 1)) - 1] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // DONE
            // This approach uses balanced trees to avoid the extra factor of log2n work
            // performed by the naive algorithm.
            // GPU Gems 3, Chapter 39.2.2

            int* dev_idata;
            // Calculate number of levels needed for the scan
            // ilog2ceil(x): computes the ceiling of log2(x), as an integer.
            int numLevels = ilog2ceil(n);
            // Calculate the power of 2 number of levels
            int numLevelsPow2 = 1 << numLevels;
            
            // Memory allocation
            cudaMalloc((void**)&dev_idata, numLevelsPow2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            int padding = numLevelsPow2 - n;
            if (padding >= 0)
            {
                cudaMemset(&dev_idata[n], 0, padding * sizeof(int));
                checkCUDAError("cudaMemset dev_idata failed!");
            }
            
            timer().startGpuTimer();

            //================================================================================
            // Part 1 - Upsweep phase
            //================================================================================
            for (int offset = 0; offset < numLevels - 1; offset++) 
            {
                // Calculate the number of blocks
                int numBlocks = (numLevelsPow2 / (1 << (offset + 1)) + blockSize - 1) / blockSize;
                
                // Perform the upsweep phase
                upSweep << <numBlocks, blockSize >> > (numLevelsPow2, offset, dev_idata);
                checkCUDAError("upSweep kernel failed!");

                // Sync before proceeding to the next iteration
                cudaDeviceSynchronize();
            }

            // Need to set the last element to 0 before starting the down sweep phase 
            cudaMemset(dev_idata + numLevelsPow2 - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");

            //================================================================================
            // Part 2 - Downsweep phase
            //================================================================================
            for (int offset = numLevels - 1; offset >= 0; offset--) {
                // Calculate the number of blocks
                int numBlocks = (numLevelsPow2 / (1 << (offset + 1)) + blockSize - 1) / blockSize;
                
                // Perform the downsweep phase
                downSweep << <numBlocks, blockSize >> > (numLevelsPow2, offset, dev_idata);
                checkCUDAError("downSweep kernel failed!");

                // Sync before proceeding to the next iteration
                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();

            // Copy the results back to the host
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy device to host (dev_idata to odata) failed!");

            // Free device memory
            cudaFree(dev_idata);
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
            // This stream compaction method will remove 0s from an array of ints.
            // Initialize necessary buffers
            int* dev_idata;
            int* dev_odata;
            int* dev_booleans;
            int* dev_indices;

            // ilog2ceil(x): computes the ceiling of log2(x), as an integer.
            int numLevels = ilog2ceil(n);
            size_t paddedSize = (size_t) 1 << numLevels;

            // Allocate memory for device arrays, copy input data to device
            cudaMalloc((void**)&dev_idata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMalloc((void**)&dev_odata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");
            cudaMalloc((void**)&dev_booleans, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_booleans failed");
            cudaMalloc((void**)&dev_indices, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed");

            if (paddedSize > n) {
                cudaMemset(dev_idata + n, 0, (paddedSize - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_idata failed!");
            }

            dim3 fullBlocksPerGrid((paddedSize + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            // Map to boolean
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (paddedSize, dev_booleans, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");

            // Perform scan on the boolean array
            cudaMemcpy(dev_indices, dev_booleans, sizeof(int) * paddedSize, cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy from dev_booleans to dev_indices (Device To Device) failed!");

            //================================================================================
            // Part 1 - Upsweep phase
            //================================================================================
            for (int offset = 0; offset < numLevels - 1; offset++) {
                
                // Calculate necessary number of blocks
                int numBlocks = (paddedSize / (1 << (offset + 1)) + blockSize - 1) / blockSize;
                
                if (numBlocks > 0) 
                {
                    upSweep << <numBlocks, blockSize >> > (paddedSize, offset, dev_indices);
                    checkCUDAError("upSweep kernel failed!");
                    cudaDeviceSynchronize();
                }
            }

            // Need to set the last element to 0 before starting the down sweep phase 
            cudaMemset(dev_indices + paddedSize - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset failed!");

            //================================================================================
            // Part 2 - Downsweep phase
            //================================================================================
            for (int offset = numLevels - 1; offset >= 0; offset--) {
                int numBlocks = (paddedSize / (1 << (offset + 1)) + blockSize - 1) / blockSize;
                if (numBlocks > 0) 
                {
                    downSweep << <numBlocks, blockSize >> > (paddedSize, offset, dev_indices);
                    checkCUDAError("downSweep kernel failed!");
                    cudaDeviceSynchronize();
                }
            }

            //================================================================================
            // Part 3: Scatter
            //================================================================================
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (paddedSize, dev_odata, dev_idata, dev_booleans, dev_indices);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            //================================================================================
            // Part 4: Copy results and free memory
            //================================================================================
            int numRemaining;
            cudaMemcpy(&numRemaining, dev_indices + paddedSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy numRemaining failed!");

            // Copy result from device to host
            cudaMemcpy(odata, dev_odata, sizeof(int) * numRemaining, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            // Free memory of temp buffers
            cudaFree(dev_booleans);
            cudaFree(dev_indices);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            return numRemaining;
        }
    }
}
