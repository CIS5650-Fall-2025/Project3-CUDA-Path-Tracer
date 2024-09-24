#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int prev = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = idata[i] + prev;
                prev = odata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int idx = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] > 0) {
                    odata[idx] = idata[i];
                    idx++;
                }
            }
            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            std::vector<int> step1;
            for (int i = 0; i < n; i++) {
                if (idata[i] > 0) {
                    step1.push_back(1);
                }
                else {
                    step1.push_back(0);
                }
            }
            std::vector<int> step2;
            int prev = 0;
            for (int i = 0; i < n; i++) {
                step2.push_back(prev);
                prev += step1[i];
            }
            int count = step2[n - 1] + (step1[n - 1] > 0);
            for (int i = 0; i < n; i++) {
                if (idata[i] > 0) {
                    odata[step2[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }
    }
}
