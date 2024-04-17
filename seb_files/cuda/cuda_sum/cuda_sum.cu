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

__global__ void gpt3_sumKernel(int *input, int size, int *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int sum = 0;

    // Calculate sum of elements by each thread
    while (tid < size) {
        sum += input[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Reduce sum across threads in a block
    __shared__ int sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void gpt4_sumArrayKernel(int *array, int *sum, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    __shared__ int cache[256];  // Adjust based on the architecture

    int temp = 0;
    while (index < n) {
        temp += array[index];
        index += stride;
    }

    // Using shared memory to reduce the results per block
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = temp;

    __syncthreads();

    // Reduction within the block
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(sum, cache[0]);
}

__global__ void gemini_find_sum(int* A, int n, int* sum_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_sum = 0; // Local thread sum

  // Check for threads exceeding valid array bounds
  if (i < n) {
    thread_sum = A[i];
  }

  // Shared memory for efficient reduction within a block
  __shared__ int shared_sum[BLOCK_SIZE];
  shared_sum[threadIdx.x] = thread_sum;
  __syncthreads();

  // Reduce shared memory array to find the block sum
  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Thread with index 0 within the block updates the global sum using atomicAdd
  if (threadIdx.x == 0) {
    atomicAdd(sum_ptr, shared_sum[0]);
  }
}

// CUDA Kernel function to sum the elements of the array
__global__ void copilot_sum(int *d_in, int *d_out, int size) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? d_in[i] : 0;

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();    // make sure all additions at one stage are done!
    }

    // only thread 0 writes result for this block back to global memory
    if (tid == 0) {
        atomicAdd(d_out, sdata[0]);
    }
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
    int h_result;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // TODO: Setup CUB stuff as needed
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(
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
            srand(std::time(nullptr));
            for (int k = 0; k < N; k++) {
                temp[k] = rand() % 1000;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(d_result, 0, sizeof(int));

            switch (i) {
                case CUB:
                    timer.start();
                    cub::DeviceReduce::Sum(
                        d_temp_storage, temp_storage_bytes, in, d_result, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case THRUST:
                    timer.start();
                    h_result = thrust::reduce(thrust::device, in, in + N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT3:
                    timer.start();
                    gpt3_sumKernel<<<num_blocks, BLOCK_SIZE>>>(in, N, d_result);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT4:
                    timer.start();
                    gpt4_sumArrayKernel<<<num_blocks, BLOCK_SIZE>>>(in, d_result, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case COPILOT:
                    timer.start();
                    copilot_sum<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(in, d_result, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GEMINI:
                    timer.start();
                    gemini_find_sum<<<num_blocks, BLOCK_SIZE>>>(in, N, d_result);
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
                cub::DeviceReduce::Sum(
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