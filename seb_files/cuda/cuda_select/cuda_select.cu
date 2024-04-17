#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/gather.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

__global__ void gpt3_gatherKernel(const int* input, const int* flag, int* output, int* count, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int validCount = 0;

    // Each thread checks its corresponding element in the flag array
    if (tid < size && flag[tid] == 1) {
        output[validCount] = input[tid];
        validCount++;
    }

    // Use atomic operation to accumulate valid count across all threads
    atomicAdd(count, validCount);
}

__global__ void gpt4_compactWithFlags(int *input, int *flags, int n, int *output, int *count) {
    extern __shared__ int temp[]; // Temporary shared array for storing flag scan results
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Step 1: Check flags and write input to temporary shared memory if flag is 1
    if (index < n && flags[index] == 1) {
        temp[index] = input[index];
    } else {
        temp[index] = 0;
    }

    __syncthreads();  // Synchronize threads to ensure all writes to temp are completed

    // Step 2: Perform an exclusive prefix sum on the flags to determine output positions
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int val = 0;
        if (index >= stride) {
            val = temp[index - stride];
        }
        __syncthreads();
        if (index >= stride) {
            temp[index] += val;
        }
        __syncthreads();
    }

    // Step 3: Write elements to the output array using their calculated positions
    if (index < n && flags[index] == 1) {
        output[temp[index] - 1] = input[index];  // Adjust index since we did exclusive scan
    }

    // Step 4: Write the count of '1's to the output count variable
    if (index == n - 1) {
        *count = temp[index] + (flags[index] == 1 ? 1 : 0);
    }
}

// CUDA Kernel function to filter the elements of the array
__global__ void copilot_filter_and_count(int *d_in, int *d_flags, int *d_out, int *d_count, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size && d_flags[idx] == 1) {
        int pos = atomicAdd(d_count, 1);
        d_out[pos] = d_in[idx];
    }
}


__global__ void gemini_filter_and_count(int* A, int* flag, int n, int* B, int* num_active) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int write_idx = 0; // Index for writing to output array (atomic)

  // Check for threads exceeding valid array bounds
  if (i < n) {
    if (flag[i] == 1) {
      // Use atomic operations for thread-safe writing to output
      B[atomicAdd(&write_idx, 1) - 1] = A[i];
      atomicAdd(num_active, 1);
    }
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
    int *d_in;
    int *d_flags;
    int *d_out;
    int *d_num_selected_out;

    // TODO: cudaMalloc as needed
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_flags, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMalloc(&d_num_selected_out, sizeof(int));

    // TODO: Setup CUB stuff as needed
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(
                        d_temp_storage, temp_storage_bytes, d_in, 
                        d_flags, d_out,d_num_selected_out, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    srand(40);
    int temp[N];
    int temp2[N];
    for (int k = 0; k < N; k++) {
        temp[k] = rand();
        temp2[k] = rand() % 2;
    }
    cudaMemcpy(d_in, temp, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, temp2, N * sizeof(int), cudaMemcpyHostToDevice);

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
            cudaDeviceSynchronize();
            switch (i) {
                case CUB:
                    timer.start();
                    cub::DeviceSelect::Flagged(
                        d_temp_storage, temp_storage_bytes, d_in, 
                        d_flags, d_out,d_num_selected_out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case THRUST:
                    timer.start();
                    thrust::gather_if(thrust::device, 
                        thrust::make_counting_iterator(0), 
                        thrust::make_counting_iterator(N),
                        d_flags,
                        d_in,
                        d_out);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT3:
                    timer.start();
                    gpt3_gatherKernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_flags, d_out, d_num_selected_out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT4:
                    timer.start();
                    gpt4_compactWithFlags<<<num_blocks, BLOCK_SIZE>>>(d_in, d_flags, N, d_out, d_num_selected_out);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case COPILOT:
                    timer.start();
                    copilot_filter_and_count<<<num_blocks, BLOCK_SIZE>>>(d_in, d_flags, d_out, d_num_selected_out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GEMINI:
                    timer.start();
                    gemini_filter_and_count<<<num_blocks, BLOCK_SIZE>>>(d_in, d_flags, N, d_out, d_num_selected_out);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
            }
            
            if (j != 0) {
                sum += timer.getElapsedTimeChrono();
            }

            // TODO: Verify results with library
            // if (i != 0) {
            //     int* test_out;
            //     int* test_out_num;
            //     cudaMalloc(&test_out, N * sizeof(int));
            //     cudaMalloc(&test_out_num, sizeof(int));
            //     cub::DeviceSelect::Flagged(
            //             d_temp_storage, temp_storage_bytes, d_in, 
            //             d_flags, d_out, d_num_selected_out, N);
            //     int res_num;
            //     cudaMemcpy(&res_num, test_out_num, sizeof(int), cudaMemcpyDeviceToHost);
            //     assertion = thrust::equal(thrust::device, d_out, d_out + res_num, test_out);
            //     cudaFree(test_out);
            //     cudaFree(test_out_num);
            //     if (!assertion) {
            //         break;
            //     }
            // }
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (nanoseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    // TODO: Free as needed
    cudaFree(d_in);
    cudaFree(d_flags);
    cudaFree(d_out);
    cudaFree(d_num_selected_out);
    cudaFree(d_temp_storage);
}