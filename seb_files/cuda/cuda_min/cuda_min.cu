#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY AN LLM!           |
 *-------------------------------*/

__global__ void gpt3_findMinKernel(int *input, int size, int *min) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : INT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] < sdata[tid]) {
            atomicMin(min, sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(min, sdata[0]);
    }
}

__global__ void gpt4_findMinKernel(int *array, int size, int *minVal) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem from global mem
    sdata[tid] = (i < size) ? array[i] : INT_MAX;
    __syncthreads();

    // Perform reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < size) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) atomicMin(minVal, sdata[0]);
}

__global__ void copilot_findMinimum(int *array, int size, int *result) {
    extern __shared__ int shared[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    shared[threadIdx.x] = (tid < size) ? array[tid] : INT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = min(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(result, shared[0]);
    }
}

__global__ void gemini_find_min(int* A, int n, int* min_value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_min = INT_MAX; // Initialize local minimum

  // Check for threads exceeding valid array bounds
  if (i < n) {
    local_min = min(local_min, A[i]);
  }

  // Use shared memory for efficient reduction within a block
  __shared__ int shared_min[BLOCK_SIZE];
  shared_min[threadIdx.x] = local_min;
  __syncthreads();

  // Reduce shared memory array to find the block minimum
  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_min[threadIdx.x] = min(shared_min[threadIdx.x], shared_min[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  // Thread with index 0 within the block updates the global minimum (atomicMin not required)
  if (threadIdx.x == 0) {
    *min_value = shared_min[0];
  }
}

__global__ void my_find_min(int* A, int n, int* min_value) {
    int min = INT_MAX;
    for (int i = 0; i < n; i++) {
        if (A[i] < min) {
            min = A[i];
        }
    }
    *min_value = min;
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Timer<std::nano> timer;
    bool assertion = true;
    std::chrono::duration<double, std::nano> sum(0);

    // TODO: Setup initial variables
    int* in;
    int* d_result;
    int* h_result;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // TODO: Setup CUB stuff as needed
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Min(
        d_temp_storage, temp_storage_bytes, in, d_result, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    for (int i = 0; i < 6; i++) {
        sum = std::chrono::duration<double, std::nano>(0);

        switch (i) {
            case CUB:
                printf("Testing CUB!\n");
                break;
            case THRUST:
                printf("Testing Thrust!\n");
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
            srand(i * j);
            for (int k = 0; k < N; k++) {
                temp[k] = rand();
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

            int min_val = INT_MAX;
            cudaMemcpy(d_result, &min_val, sizeof(int), cudaMemcpyHostToDevice);

            switch (i) {
                case CUB:
                    timer.start();
                    cub::DeviceReduce::Min(
                        d_temp_storage, temp_storage_bytes, in, d_result, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case THRUST:
                    timer.start();
                    h_result = thrust::min_element(thrust::device, in, in + N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT3:
                    timer.start();
                    gpt3_findMinKernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(in, N, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT4:
                    timer.start();
                    gpt4_findMinKernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(in, N, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case COPILOT:
                    timer.start();
                    copilot_findMinimum<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(in, N, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GEMINI:
                    timer.start();
                    gemini_find_min<<<num_blocks, BLOCK_SIZE>>>(in, N, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
            }
            
            if (j != 0) {
                sum += timer.getElapsedTimeChrono();
            }

            // TODO: Verify results with library
            if (i != 0 && i != 1) {
                int *intended_result;
                cudaMalloc(&intended_result, sizeof(int));
                cub::DeviceReduce::Min(
                        d_temp_storage, temp_storage_bytes, in, intended_result, N);
                assertion = thrust::equal(thrust::device, d_result, d_result + 1, intended_result);
                int res1;
                int res2;
                cudaMemcpy(&res1, d_result, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&res2, intended_result, sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(intended_result);
                if (!assertion) {
                    printf("\tintended_result: %d, actual result: %d\n", res2, res1);
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
    cudaFree(d_temp_storage);
}