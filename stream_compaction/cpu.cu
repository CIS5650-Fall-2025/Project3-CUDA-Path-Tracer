#include <cstdio>
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
            // DONE
            // Exclusive - Include the identity in the output
            odata[0] = 0;
            for(int k = 1; k < n; k++)
            {
                odata[k] = odata[k - 1] + idata[k - 1];
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
            
            // DONE
            int j = 0;
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] != 0)
                {
                    odata[j] = idata[i];
                    ++j;
                }
            }


            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // DONE

            int numRemaining;

            // Step 1: Compute temporary array containing
            // either 1 or 0, depending on if element meets criteria
            int* temp = new int[n];
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] == 0)
                {
                    temp[i] = 0;
                }
                else
                {
                    temp[i] = 1;
                }
            }


            // Step 2: Run exclusive scan on the temp array
            // Exclusive - Insert the identity
            odata[0] = 0;
            // Start at 1 since we inserted the identity and are shifting to the right
            for (int k = 1; k < n; ++k)
            {
                odata[k] = odata[k - 1] + temp[k - 1];
            }

            // Step 3: Scatter!
            // Result of scan is index into the final array
            numRemaining = odata[n - 1];
            for (int i = 0; i < n; ++i)
            {
                if (temp[i] == 1)
                {
                    odata[odata[i]] = idata[i];
                }
            }

            timer().endCpuTimer();

            return numRemaining;
        }
    }
}
