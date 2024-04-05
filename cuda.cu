#include <cuda.h>
#include "utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

#define BLOCK_SIZE 256

__global__ void gpt_inclusiveScan(int* input, int* output, int n) {
    __shared__ int temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Copy input to shared memory
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0;
    }

    __syncthreads();

    // Reduction phase
    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE * 2) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Post-reduction phase
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE * 2) {
            temp[index + stride] += temp[index];
        }
    }

    __syncthreads();

    // Write result to output
    if (idx < n) {
        output[idx] = temp[tid];
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

int main(int argc, char *argv[]) {
    int type;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) {
            int num = atoi(argv[i + 1]);
            if (num > 2 || num < 0) {
                printf("okay, smartass.\n");
                return 1;
            }
            type = num;
            i++;

            switch (type) {
                case 0:
                    printf("Using GPT-3!\n");
                    break;
                case 1:
                    printf("Using library call!\n");
                    break;
                case 2:
                    printf("Using GPT-4!\n");
                    break;
            }
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("./test_code <flags>\n"
                    "\t-t [num]     : Determines what type of output to use (0: GPT-3, 1: library, 2: GPT-4)\n");
            return 0;
        } else {
            printf("./test_code <flags>\n"
                    "\t-t [num]     : Determines what type of output to use (0: GPT-3, 1: library, 2: GPT-4)\n");
            return 0;
        }
    }

    double sum;
    int const NUM_TRIALS = 100;

    for (int i = 0; i < NUM_TRIALS; i++) {
        const int N = 10000;
        int *in;
        int *out;
        Timer<std::nano> timer;
        uint64_t time_taken;

        cudaMalloc(&in, N * sizeof(int));
        cudaMalloc(&out, N * sizeof(int));

        int temp[N];
        for (int j = 0; j < N; j++) {
            temp[j] = j;
        }

        cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

        int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;


        switch (type) {
            case 0:
                timer.start();
                gpt_inclusiveScan<<<numBlocks, BLOCK_SIZE>>>(in, out, N);
                timer.stop();
                break;
            case 1:
                timer.start();
                thrust::inclusive_scan(thrust::device, in, in + N, out);
                // cub::DeviceScan::InclusiveSum(nullptr, 0, in, out, N);
                timer.stop();
                break;
            case 2:
                // TODO: ChatGPT-4
                break;
        }

        time_taken = timer.getElapsedTime();

        // print_int_array(out, 100);

        cudaFree(in);
        cudaFree(out);

        sum += time_taken;
        printf("time taken for trial %d (nanoseconds): %ld\n", i, time_taken);
    }

    printf("total avg time (nanoseconds): %f\n", sum / NUM_TRIALS);
}