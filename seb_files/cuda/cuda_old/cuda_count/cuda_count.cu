#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/count.h>
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

struct CustomReduce
{
    template <typename T>
    __host__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Timer<std::nano> timer;
    bool assertion = true;

    // TODO: Setup initial variables
    int* in;
    int* d_result;
    int result;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    for (int i = 0; i < 5; i++) {
        std::chrono::duration<double, std::nano> sum(0);

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
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT3:
                    timer.start();
                    gpt_countEquals<<<num_blocks, BLOCK_SIZE>>>(in, N, 1, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    timer.start();
                    copilot_countTarget<<<num_blocks, BLOCK_SIZE>>>(in, N, 1, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GEMINI:
                    timer.start();
                    gemini_count_targets<<<num_blocks, BLOCK_SIZE>>>(in, N, 1, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
            }
            
            if (j != 0) {
                sum += timer.getElapsedTimeChrono();
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
        printf("\ttotal avg time (nanoseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    // TODO: Free as needed
    cudaFree(d_result);
    cudaFree(in);
}