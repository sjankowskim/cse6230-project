#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY AN LLM!           |
 *-------------------------------*/

#define BLOCK_SIZE 256

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
    int const NUM_TRIALS = 1000;
    const int N = 100000;
    bool assertion = true;

    // TODO: Setup initial variables
    int* in;
    int* out;

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
            int temp[N];
            srand(j * i);
            for (int k = 0; k < N; k++) {
                temp[k] = rand() % 1000;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

            int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer.start();
            switch (i) {
                case LIBRARY:
                    thrust::copy(thrust::device, in, in + N, out);
                    break;
                case GPT3:
                    gpt_copyArray<<<num_blocks, BLOCK_SIZE>>>(in, in + N, out);
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    copilot_array_copy<<<num_blocks, BLOCK_SIZE>>>(in, out, N);
                    break;
                case GEMINI:
                    gemini_copy_array<<<num_blocks, BLOCK_SIZE>>>(in, out, N);
                    break;
            }
            timer.stop();
            if (j != 0) {
                sum += timer.getElapsedTime();
            }

            // TODO: Verify results with library
            if (i != 0) {
                int* d_temp_out;
                cudaMalloc(&d_temp_out, N * sizeof(int));
                thrust::copy(thrust::device, in, in + N, d_temp_out);
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