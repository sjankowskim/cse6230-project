#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <assert.h>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY AN LLM!           |
 *-------------------------------*/

#define BLOCK_SIZE (256)
#define ARRAY_SIZE (1000)
#define TOTAL_SIZE (2000)

__global__ void gpt_mergeSortedArrays(int* array1, int* array2, int* mergedArray) {
    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;

    int startIdx1 = blockId * blockSize;
    int startIdx2 = startIdx1 + ARRAY_SIZE;

    int endIdx1 = min(startIdx1 + blockSize, ARRAY_SIZE);
    int endIdx2 = min(startIdx2 + blockSize, TOTAL_SIZE);

    int idx1 = startIdx1;
    int idx2 = startIdx2;

    int mergedIdx = startIdx1 + tid * 2;

    while (idx1 < endIdx1 && idx2 < endIdx2) {
        if (array1[idx1] <= array2[idx2 - ARRAY_SIZE]) {
            mergedArray[mergedIdx] = array1[idx1];
            idx1++;
        } else {
            mergedArray[mergedIdx] = array2[idx2 - ARRAY_SIZE];
            idx2++;
        }
        mergedIdx++;
    }

    while (idx1 < endIdx1) {
        mergedArray[mergedIdx] = array1[idx1];
        idx1++;
        mergedIdx++;
    }

    while (idx2 < endIdx2) {
        mergedArray[mergedIdx] = array2[idx2 - ARRAY_SIZE];
        idx2++;
        mergedIdx++;
    }
}

__device__ int binarySearch(int* array, int start, int end, int value) {
    while (start <= end) {
        int mid = start + (end - start) / 2;
        if (array[mid] < value)
            start = mid + 1;
        else if (array[mid] > value)
            end = mid - 1;
        else
            return mid;
    }
    return -1;
}

__global__ void copilot_mergeSortedArrays(int* d_a, int* d_b, int* d_c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        int ai = i;
        int bi = binarySearch(d_b, 0, n - 1, d_a[ai]);
        if (bi != -1) {
            d_c[2*ai] = d_a[ai];
            d_c[2*ai + 1] = d_b[bi];
        }
    }
}

__global__ void gemini_merge(int* A, int* B, int* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID within block
  int stride = blockDim.x; // Thread stride for coalesced memory access

  // Loop through all elements that this thread is responsible for
  for (int k = i; k < 2*n; k += stride) {
    int a_idx = k < n ? k : (k - n); // Index for A (bounded by n)
    int b_idx = k >= n ? (k - n) : 0;   // Index for B (bounded by n)

    // Choose element from A or B based on sorting order and boundary checks
    C[k] = (a_idx < n && A[a_idx] <= B[b_idx]) ? A[a_idx] : B[b_idx]; 
  }
}


/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

bool arraysEqual(int* arr1, int* arr2, int size) {
    for (int i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

int compare(const void *a, const void *b) {
    int int_a = *((int*)a);
    int int_b = *((int*)b);
    if (int_a < int_b) return -1;
    if (int_a > int_b) return 1;
    return 0;
}

int main() {
    Timer<std::nano> timer;
    int const NUM_TRIALS = 1000;
    const int N = 100000;
    bool assertion = true;

    // TODO: Setup initial variables
    int *in_1;
    int *in_2;
    int *result;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in_1, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&in_2, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&result, TOTAL_SIZE * sizeof(int));

    for (int i = 0; i < 5; i++) {
        double sum = 0;

        switch (i) {
            case LIBRARY:
                printf("Testing library call!\n");
                break;
            case GPT3:
                printf("Testing GPT-3.5!\n");
                break;
            case GPT4:
                printf("Testing GPT-4!\n");
                break;
            case COPILOT:
                printf("Testing Copilot!\n");
                break;
            case GEMINI:
                printf("Testing Gemini!\n");
                break;
        }

        for (int j = 0; j < NUM_TRIALS; j++) {

            // TODO: Setup initial variables and cudaMemcpy as needed.
            srand(j);
            int temp_1[ARRAY_SIZE];
            int temp_2[ARRAY_SIZE];
            for (int k = 0; k < ARRAY_SIZE; k++) {
                temp_1[k] = rand() % 999 + 1;
                temp_2[k] = rand() % 999 + 1;
            }
            qsort(temp_1, ARRAY_SIZE, sizeof(int), compare);
            qsort(temp_2, ARRAY_SIZE, sizeof(int), compare);
            cudaMemcpy(in_1, temp_1, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(in_2, temp_2, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(result, 0, TOTAL_SIZE * sizeof(int));

            int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer.start();
            switch (i) {
                case LIBRARY:
                    thrust::merge(thrust::device, in_1, in_1 + ARRAY_SIZE, in_2, in_2 + ARRAY_SIZE, result);
                    break;
                case GPT3:
                    gpt_mergeSortedArrays<<<num_blocks, BLOCK_SIZE>>>(in_1, in_2, result);
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    copilot_mergeSortedArrays<<<num_blocks, BLOCK_SIZE>>>(in_1, in_2, result, ARRAY_SIZE);
                    break;
                case GEMINI:
                    gemini_merge<<<num_blocks, BLOCK_SIZE>>>(in_1, in_2, result, ARRAY_SIZE);
                    break;
            }
            timer.stop();
            if (j != 0) {
                sum += timer.getElapsedTime();
            }

            // TODO: Verify results with library
            if (i != 0) {
                int* lib_result;
                cudaMalloc(&lib_result, TOTAL_SIZE * sizeof(int));
                thrust::merge(thrust::device, in_1, in_1 + ARRAY_SIZE, in_2, in_2 + ARRAY_SIZE, lib_result);
                int* h_res1 = (int*) malloc(TOTAL_SIZE * sizeof(int));
                int* h_res2 = (int*) malloc(TOTAL_SIZE * sizeof(int));

                if (!h_res1 || !h_res2) {
                    free(h_res1);
                    free(h_res2);
                    printf("malloc failed while asserting!\n");
                    return 1;
                }
                
                cudaMemcpy(h_res1, result, TOTAL_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_res2, lib_result, TOTAL_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(lib_result);
                assertion = arraysEqual(h_res1, h_res2, TOTAL_SIZE);
                if (!assertion) {
                    free(h_res1);
                    free(h_res2);
                    break;
                }
                free(h_res1);
                free(h_res2);
            }
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (nanoseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    // TODO: Free as needed
    cudaFree(in_1);
    cudaFree(in_2);
    cudaFree(result);
}