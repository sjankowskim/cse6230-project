#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY AN LLM!           |
 *-------------------------------*/

__global__ void gpt_countEquals(int *array, int n, int target, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (array[i] == target) {
            count++;
        }
    }

    atomicAdd(result, count);
}

__global__ void copilot_countTarget(int *d_array, int n, int target, int *d_count) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n && d_array[index] == target) {
        atomicAdd(d_count, 1);
    }
}


__global__ void gemini_count_targets(int* A, int n, int target, int* partial_counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;

  // Loop through elements assigned to this thread
  for (int j = i; j < n; j += blockDim.x) {
    sum += (A[j] == target);
  }

  // Use shared memory for efficient thread-local reduction
  __shared__ int shared_count[256];
  shared_count[threadIdx.x] = sum;
  __syncthreads();

  // Reduce shared memory array using warp shuffle and reduction
  int tid = threadIdx.x;
  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    if (tid < stride) {
      shared_count[tid] += shared_count[tid + stride];
    }
    __syncthreads();
  }

  // Final reduction using atomicAdd for thread-safe accumulation
  if (tid == 0) {
    atomicAdd(partial_counts, shared_count[0]);
  }
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Timer2<std::milli> timer;
    bool assertion = true;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // TODO: Setup initial variables
    int* in;
    int* d_result;
    int result;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

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
            int temp[N];
            srand(j * i);
            for (int k = 0; k < N; k++) {
                temp[k] = rand() % 10;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(d_result, 0, sizeof(int));
            
            switch (i) {
                case LIBRARY:
                    timer.start();
                    result = thrust::count(thrust::device, in, in + N, 1);
                    timer.stop();
                    break;
                case GPT3:
                    cudaEventRecord(start, 0);
                    gpt_countEquals<<<num_blocks, BLOCK_SIZE>>>(in, N, 1, d_result);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    cudaEventRecord(start, 0);
                    copilot_countTarget<<<num_blocks, BLOCK_SIZE>>>(in, N, 1, d_result);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    break;
                case GEMINI:
                    cudaEventRecord(start, 0);
                    gemini_count_targets<<<num_blocks, BLOCK_SIZE>>>(in, N, 1, d_result);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    break;
            }
            
            if (j != 0) {
                if (i == 0){
                    sum += timer.getElapsedTime();
                } else {
                    float elapsedTime;
                    cudaEventElapsedTime(&elapsedTime, start, stop);
                    sum += elapsedTime;
                }
            }

            // TODO: Verify results with library
            if (i != 0) {
                int intended_result = thrust::count(thrust::device, in, in + N, 1);
                cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
                assertion = (result == intended_result);
                if (!assertion) {
                    printf("\tintended_result: %d, actual result: %d\n", intended_result, result);
                    break;
                }
            }
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (milliseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // TODO: Free as needed
    cudaFree(d_result);
    cudaFree(in);
}