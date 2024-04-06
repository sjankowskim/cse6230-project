#include <cuda.h>
#include "../../utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cassert>
#include <cstdlib>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

#define BLOCK_SIZE (256)
#define ARRAY_SIZE (1000)
#define TOTAL_SIZE (2000)

__global__ void gpt_mergeAndSort(int* array1, int* array2, int* mergedArray) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;

    int startIdx1 = bid * blockSize;
    int startIdx2 = startIdx1 + ARRAY_SIZE;
    int endIdx1 = startIdx2;
    int endIdx2 = startIdx2 + blockSize;

    int idx1 = startIdx1 + tid;
    int idx2 = startIdx2 + tid;

    __shared__ int tempArray[TOTAL_SIZE];

    // Copy elements from array1 and array2 to shared memory
    if (idx1 < endIdx1) {
        tempArray[idx1] = array1[idx1];
    }
    if (idx2 < endIdx2) {
        tempArray[idx2] = array2[idx2 - ARRAY_SIZE];
    }

    __syncthreads();

    // Perform merge sort within shared memory
    int i = startIdx1, j = startIdx2, k = startIdx1;
    while (i < endIdx1 && j < endIdx2) {
        if (tempArray[i] <= tempArray[j]) {
            mergedArray[k++] = tempArray[i++];
        } else {
            mergedArray[k++] = tempArray[j++];
        }
    }
    while (i < endIdx1) {
        mergedArray[k++] = tempArray[i++];
    }
    while (j < endIdx2) {
        mergedArray[k++] = tempArray[j++];
    }
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

int main() {
    double sum;
    int *in_1;
    int *in_2;
    int *result;
    int const NUM_TRIALS = 1000;

    Timer<std::nano> timer;
    uint64_t time_taken;

    cudaMalloc(&in_1, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&in_2, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&result, TOTAL_SIZE * sizeof(int));

    for (int i = 0; i < 2; i++) {
        sum = 0;
        for (int j = 0; j < NUM_TRIALS; j++) {
            srand(j);
            int temp_1[ARRAY_SIZE];
            int temp_2[ARRAY_SIZE];
            for (int k = 0; k < ARRAY_SIZE; k++) {
                temp_1[k] = rand();
                temp_2[k] = rand();
            }
            cudaMemcpy(in_1, temp_1, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(in_2, temp_2, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

            int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

            switch (i) {
                case 0:
                    timer.start();
                    gpt_mergeAndSort<<<numBlocks, BLOCK_SIZE>>>(in_1, in_2, result);
                    timer.stop();
                    break;
                case 1:
                    timer.start();
                    thrust::merge(thrust::device, in_1, in_1 + ARRAY_SIZE, in_2, in_2 + ARRAY_SIZE, result);
                    timer.stop();
                    break;
                case 2:
                    // TODO: ChatGPT-4
                    break;
            }

            if (i == 0) {
                int* lib_result;
                cudaMalloc(&lib_result, TOTAL_SIZE * sizeof(int));
                thrust::merge(thrust::device, in_1, in_1 + ARRAY_SIZE, in_2, in_2 + ARRAY_SIZE, lib_result);
                int h_res1[TOTAL_SIZE];
                int h_res2[TOTAL_SIZE];
                cudaMemcpy(h_res1, result, TOTAL_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_res2, lib_result, TOTAL_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
                assert(arraysEqual(h_res1, hres_2, TOTAL_SIZE));
                cudaFree(lib_result);
            }

            time_taken = timer.getElapsedTime();

            sum += time_taken;
        }

        switch (i) {
            case 0:
                printf("Testing GPT-3.5!\n");
                break;
            case 1:
                printf("Testing library call!\n");
                break;
            case 2:
                // TODO: ChatGPT-4
                break;
        }

        printf("total avg time (nanoseconds): %f\n", sum / NUM_TRIALS);
    }

    cudaFree(in_1);
    cudaFree(in_2);
    cudaFree(result);
}