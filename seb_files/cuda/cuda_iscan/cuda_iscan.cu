#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

__global__ void gpt3_prefixSumKernel(int *input, int *output, int size) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;

    // Load input into shared memory.
    sdata[tid] = (tid < size) ? input[tid] : 0;
    __syncthreads();

    // Build sum in place up the tree
    for (int offset = 1; offset < size; offset *= 2) {
        int temp;
        if (tid >= offset) {
            temp = sdata[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            sdata[tid] += temp;
        }
        __syncthreads();
    }

    // Write the result to output array
    if (tid < size) {
        output[tid] = sdata[tid];
    }
}

__global__ void gpt4_inclusive_scan_kernel(int *input, int *output, int n) {
    extern __shared__ int temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;

    int ai = thid;
    int bi = thid + (n/2);

    // Load input into shared memory.
    // This is assuming the block size is at least half of n
    temp[ai] = (ai < n) ? input[ai] : 0;
    temp[bi] = (bi < n) ? input[bi] : 0;

    // Build the sum in place up the tree
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) { temp[n - 1] = 0; } // clear the last element

    // Traverse down tree & build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to device memory
    if (ai < n) output[ai] = temp[ai];
    if (bi < n) output[bi] = temp[bi];
}

__global__ void gemini_prefix_sum(int* A, int* B, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_sum = 0; // Local thread sum

  // Calculate prefix sum for the first element efficiently (no reduction)
  if (i == 0) {
    thread_sum = A[0];
  } else {
    // Reduction within a warp to efficiently calculate prefix sum for remaining elements
    __shared__ int shared_sum[32];  // Shared memory for warp-level reduction
    int offset = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    if (lane_idx == 0) {
      shared_sum[lane_idx] = (i < n) ? A[i] : 0;
    }
    __syncthreads();

    for (int stride = 1; stride <= 16; stride *= 2) {
      if (lane_idx < stride) {
        shared_sum[lane_idx] += shared_sum[lane_idx + stride];
      }
      __syncthreads();
    }

    // Final accumulation considering warp offset
    thread_sum = (offset * 32 + lane_idx < n) ? shared_sum[lane_idx] : 0;
  }

  // Add prefix sum from previous block (scan approach)
  if (i > 0) {
    thread_sum += B[i - 1];
  }

  // Store the result in the output array
  if (i < n) {
    B[i] = thread_sum;
  }
}

__global__ void copilot_prefixSum(int *input, int *output, int size) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;

    // Load input into shared memory.
    temp[2*tid] = (2*tid < size) ? input[2*tid] : 0;
    temp[2*tid+1] = (2*tid+1 < size) ? input[2*tid+1] : 0;
    __syncthreads();

    // Up-sweep phase.
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (tid+1)*stride*2 - 1;
        if (index < 2*blockDim.x) {
            temp[index] += temp[index-stride];
        }
        __syncthreads();
    }

    // Down-sweep phase.
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid+1)*stride*2 - 1;
        if (index + stride < 2*blockDim.x) {
            temp[index+stride] += temp[index];
        }
    }
    __syncthreads();

    // Write the output.
    if (2*tid < size) output[2*tid] = temp[2*tid];
    if (2*tid+1 < size) output[2*tid+1] = temp[2*tid+1];
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
    int *in;
    int *out;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&out, N * sizeof(int));

    // TODO: Setup CUB stuff as needed
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
                        d_temp_storage, temp_storage_bytes, in, out, N);
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
            srand(j * i);
            int temp[N];
            for (int k = 0; k < N; k++) {
                temp[k] = rand() % 1000;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

            switch (i) {
                case CUB:
                    timer.start();
                    cub::DeviceScan::InclusiveSum(
                        d_temp_storage, temp_storage_bytes, in, out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case THRUST:
                    timer.start();
                    thrust::inclusive_scan(thrust::device, in, in + N, out);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT3:
                    timer.start();
                    gpt3_prefixSumKernel<<<1, N, N * sizeof(int)>>>(in, out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT4:
                    // DID NOT WORK, CRASHES PROGRAM!
                    // timer.start();
                    // gpt4_inclusive_scan_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(in, out, N);
                    // cudaDeviceSynchronize();
                    // timer.stop();
                    break;
                case COPILOT:
                    timer.start();
                    copilot_prefixSum<<<num_blocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(int)>>>(in, out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GEMINI:
                    timer.start();
                    gemini_prefix_sum<<<num_blocks, BLOCK_SIZE>>>(in, out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
            }
            
            if (j != 0) {
                sum += timer.getElapsedTimeChrono();
            }

            // TODO: Verify results with library
            if (i != 0 && i != 1) {
                int* test_out;
                cudaMalloc(&test_out, N * sizeof(int));
                cub::DeviceScan::InclusiveSum(
                        d_temp_storage, temp_storage_bytes, in, test_out, N);
                assertion = thrust::equal(thrust::device, out, out + N, test_out);
                cudaFree(test_out);
                if (!assertion) {
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
    cudaFree(out);
    cudaFree(in);
    cudaFree(d_temp_storage);
}