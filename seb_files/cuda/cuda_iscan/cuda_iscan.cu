#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

#define BLOCK_SIZE 256

__global__ void gpt_inclusiveScan(int* input, int* output, int n) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        // Perform inclusive scan
        int sum = 0;
        for (int i = 0; i <= tid; ++i) {
            sum += input[blockIdx.x * blockDim.x + i];
        }
        output[index] = sum;
    }
}

__global__ void gemini_inclusive_scan(int* A, int* B, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_sum = 0; // Local variable for thread's partial sum

  // Prefix scan within a warp using shared memory
  __shared__ int shared_scan[256];
  if (threadIdx.x < n) {
    shared_scan[threadIdx.x] = A[threadIdx.x];
  } else {
    shared_scan[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_scan[threadIdx.x] += shared_scan[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Final reduction using atomicAdd for thread-safe accumulation
  if (threadIdx.x < n) {
    thread_sum = atomicAdd(&B[i], shared_scan[threadIdx.x]);
  }

  // Exclusive prefix sum within a warp for efficient propagation (modified)
  __shared__ int shared_exclusive_scan[256];
  shared_exclusive_scan[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      int temp = shared_exclusive_scan[threadIdx.x];
      shared_exclusive_scan[threadIdx.x] += shared_exclusive_scan[threadIdx.x + stride];
      shared_exclusive_scan[threadIdx.x + stride] = temp; // Swap for correct propagation
    }
    __syncthreads();
  }

  // Add prefix sum from previous block and store the final scan result
  if (i > 0) {
    B[i] = thread_sum + shared_exclusive_scan[threadIdx.x - 1];
  } else { // Special case for the first element (no previous block sum)
    B[i] = thread_sum;
  }
}

__global__ void copilot_scan_kernel(int *g_odata, int *g_idata, int n) {
    extern __shared__ int temp[]; // allocated on invocation

    int thid = threadIdx.x;
    int offset = 1;

    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];

    // Build sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Clear the last element
    if (thid == 0) { temp[n - 1] = 0; }

    // Traverse down the tree
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results to device memory
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

void print_int_array(int* arr, int size) {
    int* temp = (int *) malloc(size * sizeof(int));
    if (temp == 0) {
        printf("malloc failed, ruh roh!\n");
        return;
    }
    cudaMemcpy(temp, arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("----------------------\n");
    for (int i = 0; i < size; i++) {
        printf("[%d]: %d\n", i, temp[i]);
    }

    free(temp);
}

bool array_equals(int* arr1, int* arr2, int n) {
    for (int i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    Timer<std::nano> timer;
    uint64_t time_taken;
    double sum;
    int const NUM_TRIALS = 1000;
    const int N = 100000;
    bool assertion = true;

    // TODO: Setup initial variables
    int *in;
    int *out;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&out, N * sizeof(int));

    for (int i = 0; i < 5; i++) {
        sum = 0;

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

            // TODO: Setup initial array and cudaMemcpy as needed.
            // Reset any variables as needed
            srand(j * i);
            int temp[N];
            for (int k = 0; k < N; k++) {
                temp[k] = rand() % 999;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

            int num_blocks;
            timer.start();
            switch (i) {
                case LIBRARY:
                    thrust::inclusive_scan(thrust::device, in, in + N, out);
                    break;
                case GPT3:
                    gpt_inclusiveScan<<<1, BLOCK_SIZE>>>(in, out, N);
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    copilot_scan_kernel<<<1, N/2, N * sizeof(int)>>>(in, out, N);
                    break;
                case GEMINI:
                    num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    gemini_inclusive_scan<<<num_blocks, BLOCK_SIZE>>>(in, out, N);
                    break;
            }
            timer.stop();
            time_taken = timer.getElapsedTime();

            // TODO: Verify results with library
            if (i != 0) {
                int* test_out;
                cudaMalloc(&test_out, N * sizeof(int));

                int* llm_arr = (int*) malloc(N * sizeof(int));
                int* test_arr = (int*) malloc(N * sizeof(int));

                if (!llm_arr || !test_arr) {
                    free(llm_arr);
                    free(test_arr);
                    printf("malloc failed!\n");
                    return 1;
                }

                thrust::inclusive_scan(thrust::device, in, in + N, test_out);
                cudaMemcpy(llm_arr, out, N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(test_arr, test_out, N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(test_out);
                assertion = array_equals(test_arr, llm_arr, N); 
                if (!assertion) {
                    // for (int k = 0; k < N; k++) {
                    //     printf("llm_arr[%d]: %d, \t test_arr[%d]: %d\n", k, llm_arr[k], k, test_arr[k]);
                    // }
                    free(llm_arr);
                    free(test_arr);
                    break;
                }
                free(llm_arr);
                free(test_arr);
            }
            
            sum += time_taken;
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (nanoseconds): %f\n", sum / NUM_TRIALS);
    }

    // TODO: Free as needed
    cudaFree(in);
    cudaFree(out);
}

int main() {
    Timer<std::nano> timer;
    int const NUM_TRIALS = 1000;
    const int N = 100000;
    bool assertion = true;

    // TODO: Setup initial variables
    int *in;
    int *out;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&out, N * sizeof(int));

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
            srand(j * i);
            int temp[N];
            for (int k = 0; k < N; k++) {
                temp[k] = rand() % 1000;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

            int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer.start();
            switch (i) {
                case LIBRARY:
                    thrust::inclusive_scan(thrust::device, in, in + N, out);
                    break;
                case GPT3:
                    gpt_inclusiveScan<<<1, BLOCK_SIZE>>>(in, out, N);
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    copilot_scan_kernel<<<1, N/2, N * sizeof(int)>>>(in, out, N);
                    break;
                case GEMINI:
                    gemini_inclusive_scan<<<num_blocks, BLOCK_SIZE>>>(in, out, N);
                    break;
            }
            timer.stop();
            if (j != 0) {
                sum += timer.getElapsedTime();
            }

            // TODO: Verify results with library
            if (i != 0) {
                int* test_out;
                cudaMalloc(&test_out, N * sizeof(int));

                int* llm_arr = (int*) malloc(N * sizeof(int));
                int* test_arr = (int*) malloc(N * sizeof(int));

                if (!llm_arr || !test_arr) {
                    free(llm_arr);
                    free(test_arr);
                    printf("malloc failed!\n");
                    return 1;
                }

                thrust::inclusive_scan(thrust::device, in, in + N, test_out);
                cudaMemcpy(llm_arr, out, N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(test_arr, test_out, N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(test_out);
                assertion = array_equals(test_arr, llm_arr, N); 
                if (!assertion) {
                    // for (int k = 0; k < N; k++) {
                    //     printf("llm_arr[%d]: %d, \t test_arr[%d]: %d\n", k, llm_arr[k], k, test_arr[k]);
                    // }
                    free(llm_arr);
                    free(test_arr);
                    break;
                }
                free(llm_arr);
                free(test_arr);
            }
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (nanoseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    // TODO: Free as needed
    cudaFree(out);
    cudaFree(in);
}