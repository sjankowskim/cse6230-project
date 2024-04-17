#include <cuda.h>
#include <cub/cub.cuh>
#include "../../../utils.hpp"

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY AN LLM!           |
 *-------------------------------*/

__global__ void gpt_copyArray(const int* input_begin, const int* input_end, int* output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int* ptr = input_begin + tid;

    while (ptr < input_end) {
        output[tid] = *ptr;
        ptr += blockDim.x * gridDim.x;  // Move to the next block's element
        tid += blockDim.x * gridDim.x;  // Update thread ID for the next iteration
    }
}

__global__ void copilot_array_copy(int* d_in, int* d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = d_in[idx];
    }
}

__global__ void gemini_copy_array(const int* A, int* B, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check for threads exceeding array bounds
  if (i < n) {
    B[i] = A[i];
  }
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

bool array_equals(int* arr1, int* arr2, int n) {
    for (int i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Timer<std::nano> timer;
    bool assertion = true;

    // TODO: Setup initial variables
    int* in;
    int* out;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&out, N * sizeof(int));

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
                temp[k] = rand() % 1000;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

            switch (i) {
                case LIBRARY:
                    timer.start();
                    // TODO: thrust call
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT3:
                    timer.start();
                    gpt_copyArray<<<num_blocks, BLOCK_SIZE>>>(in, in + N, out);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    timer.start();
                    copilot_array_copy<<<num_blocks, BLOCK_SIZE>>>(in, out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GEMINI:
                    timer.start();
                    gemini_copy_array<<<num_blocks, BLOCK_SIZE>>>(in, out, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
            }
            
            if (j != 0) {
                sum += timer.getElapsedTimeChrono();
            }

            // TODO: Verify results with library
            if (i != 0) {
                int* d_temp_out;
                cudaMalloc(&d_temp_out, N * sizeof(int));
                // TODO: Thrust call
                // cub::DeviceCopy::Batched(d_temp_out, in, N);
                int h_temp_out[N];
                int h_llm_out[N];
                cudaMemcpy(h_temp_out, d_temp_out, N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_llm_out, out, N * sizeof(int), cudaMemcpyDeviceToHost);
                assertion = array_equals(h_temp_out, h_llm_out, N);
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
}