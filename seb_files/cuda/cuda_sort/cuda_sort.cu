#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY AN LLM!           |
 *-------------------------------*/

#define BLOCK_SIZE 256

__device__ void merge(int* array, int left, int mid, int right) {
    int* temp = new int[right - left + 1];
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (array[i] <= array[j]) {
            temp[k++] = array[i++];
        } else {
            temp[k++] = array[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = array[i++];
    }

    while (j <= right) {
        temp[k++] = array[j++];
    }

    for (i = left, k = 0; i <= right; ++i, ++k) {
        array[i] = temp[k];
    }

    delete[] temp;
}

// Kernel to perform merge sort
__global__ void mergeSort(int* array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int segment_size = 1;

    // Bottom-up merge sort
    while (segment_size < size) {
        int left = tid * segment_size * 2;
        int mid = left + segment_size - 1;
        int right = min(left + segment_size * 2 - 1, size - 1);

        if (mid < size) {
            merge(array, left, mid, right);
        }

        segment_size *= 2;
        __syncthreads();
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
    int const NUM_TRIALS = 2;
    const int N = 100000;
    bool assertion = true;

    // TODO: Setup initial variables
    int* in;
    int* in2;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&in2, N * sizeof(int));

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
            cudaMemcpy(in2, temp, N * sizeof(int), cudaMemcpyHostToDevice);

            int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer.start();
            switch (i) {
                case LIBRARY:
                    thrust::sort(thrust::device, in, in + N);
                    break;
                case GPT3:
                    mergeSort<<<num_blocks, BLOCK_SIZE>>>(in, N);
                    break;
                case GPT4:
                    // TODO: GPT4
                    break;
                case COPILOT:
                    break;
                case GEMINI:
                    break;
            }
            timer.stop();
            if (j != 0) {
                sum += timer.getElapsedTime();
                // printf("time for trial %d: %f\n", j, timer.getElapsedTime());
            }

            // TODO: Verify results with library
            if (i != 0) {
                thrust::sort(thrust::device, in2, in2 + N);
                // int* temp1 = (int*) malloc(N * sizeof(int));
                // int* temp2 = (int*) malloc(N * sizeof(int));
                // if (!temp1 || !temp2) {
                //     free(temp1);
                //     free(temp2);
                //     printf("malloc failed!\n");
                //     return 1;
                // }
                // cudaMemcpy(temp1, in, N * sizeof(int), cudaMemcpyDeviceToHost);
                // cudaMemcpy(temp2, in2, N * sizeof(int), cudaMemcpyDeviceToHost);
                // assertion = array_equals(temp1, temp2, N);
                // if (!assertion) {
                //     free(temp1);
                //     free(temp2);
                //     break;
                // }
                // free(temp1);
                // free(temp2);
            }
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (nanoseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    // TODO: Free as needed
    cudaFree(in2);
    cudaFree(in);
}