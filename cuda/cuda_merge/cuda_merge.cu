#include <cuda.h>
#include "../../utils.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

#define BLOCK_SIZE 256
#define ARRAY_SIZE 1000
#define TOTAL_SIZE ARRAY_SIZE << 1

__global__ void gpt_mergeAndSort(int* array1, int* array2, int* mergedArray) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;

    int startIdx1 = bid * blockSize;
    int startIdx2 = startIdx1 + ARRAY_SIZE;
    int endIdx1 = startIdx2;
    int endIdx2 = startIdx2 + blockSize;

    int idx1 = startIdx1 + tid;
    int idx2 = startIdx2 + tid;

    __shared__ int tempArray[TOTAL_SIZE];

    // Copy elements from array1 and array2 to shared memory
    if (idx1 < endIdx1) {
        tempArray[idx1] = array1[idx1];
    }
    if (idx2 < endIdx2) {
        tempArray[idx2] = array2[idx2 - ARRAY_SIZE];
    }

    __syncthreads();

    // Perform merge sort within shared memory
    int i = startIdx1, j = startIdx2, k = startIdx1;
    while (i < endIdx1 && j < endIdx2) {
        if (tempArray[i] <= tempArray[j]) {
            mergedArray[k++] = tempArray[i++];
        } else {
            mergedArray[k++] = tempArray[j++];
        }
    }
    while (i < endIdx1) {
        mergedArray[k++] = tempArray[i++];
    }
    while (j < endIdx2) {
        mergedArray[k++] = tempArray[j++];
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
        int *in_1;
        int *in_2;
        int *result;
        Timer<std::nano> timer;
        uint64_t time_taken;

        cudaMalloc(&in_1, ARRAY_SIZE * sizeof(int));
        cudaMalloc(&in_2, ARRAY_SIZE * sizeof(int));
        cudaMalloc(&result, TOTAL_SIZE * sizeof(int));

        int temp_1[ARRAY_SIZE];
        int temp_2[ARRAY_SIZE];
        for (int j = 0; j < ARRAY_SIZE; j++) {
            temp_1[j] = j;
            temp_2[j] = ARRAY_SIZE - j;
        }
        cudaMemcpy(in_1, temp_1, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(in_2, temp_2, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

        switch (type) {
            case 0:
                timer.start();
                gpt_mergeAndSort<<<numBlocks, BLOCK_SIZE>>>(in_1, in_2, result);
                timer.stop();
                break;
            case 1:
                timer.start();
                thrust::merge(thrust::device, in_1, in_1 + ARRAY_SIZE, in_2, in_2 + ARRAY_SIZE, result);
                timer.stop();
                break;
            case 2:
                // TODO: ChatGPT-4
                break;
        }

        time_taken = timer.getElapsedTime();

        // print_int_array(result, 100);

        cudaFree(in_1);
        cudaFree(in_2);
        cudaFree(result);

        sum += time_taken;
        printf("time taken for trial %d (nanoseconds): %ld\n", i, time_taken);
    }

    printf("total avg time (nanoseconds): %f\n", sum / NUM_TRIALS);
}