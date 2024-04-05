#include <cuda.h>
#include "../../utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

#define BLOCK_SIZE 256

__global__
void gpt_countOnes(const int* array, int size, int* result) {
    __shared__ int partialCounts[256]; // Shared memory for storing partial counts
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int chunkSize = (size + gridDim.x - 1) / gridDim.x; // Chunk size for each block
    int start = blockIdx.x * chunkSize;
    int end = min(start + chunkSize, size);

    // Count occurrences of 1 within the chunk
    int count = 0;
    for (int i = start + tid; i < end; i += blockSize) {
        if (array[i] == 1) {
            count++;
        }
    }

    // Store partial count in shared memory
    partialCounts[tid] = count;
    __syncthreads();

    // Perform block-level reduction using warp shuffle
    for (int offset = blockSize / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            partialCounts[tid] += partialCounts[tid + offset];
        }
        __syncthreads();
    }

    // Store block-level count in global memory
    if (tid == 0) {
        atomicAdd(result, partialCounts[0]);
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
    int const NUM_TRIALS = 1000;

    for (int i = 0; i < NUM_TRIALS; i++) {
        const int N = 1000000;
        int *in;
        Timer<std::nano> timer;
        uint64_t time_taken;
        int result = 0;

        cudaMalloc(&in, N * sizeof(int));

        int temp[N];
        for (int j = 0; j < N; j++) {
            temp[j] = j % 2;
        }

        cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

        int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        switch (type) {
            case 0:
                int *d_result;
                cudaMalloc(&d_result, sizeof(int));
                timer.start();
                gpt_countOnes<<<numBlocks, BLOCK_SIZE>>>(in, N, d_result);
                timer.stop();
                cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(d_result);
                break;
            case 1:
                timer.start();
                result = thrust::count(thrust::device, in, in + N, 1);
                timer.stop();
                break;
            case 2:
                // TODO: ChatGPT-4
                break;
        }

        time_taken = timer.getElapsedTime();

        // printf("count: %d\n", result);

        cudaFree(in);

        sum += time_taken;
        printf("time taken for trial %d (nanoseconds): %ld\n", i, time_taken);
    }

    printf("total avg time (nanoseconds): %f\n", sum / NUM_TRIALS);
}