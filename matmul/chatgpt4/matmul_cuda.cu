#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "utils.hpp"

#include <iostream>

#include <cuda_runtime.h>

/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/
using namespace std;

__global__ void matrixMultiplyCUDA(int *A, int *B, int *C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        int sum = 0;
        for (int i = 0; i < numAColumns; i++) {
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}
/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

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

    uint64_t time_taken;
    double sum = 0;
    int i = 0;

const int numARows = 100, numAColumns = 100, numBRows = 100, numBColumns = 100;

    // Calculate the size of each matrix
    int sizeA = numARows * numAColumns * sizeof(int);
    int sizeB = numBRows * numBColumns * sizeof(int);
    int sizeC = numARows * numBColumns * sizeof(int);

    // Allocate space for host matrices A, B, and C
    int *h_A = new int[numARows * numAColumns];
    int *h_B = new int[numBRows * numBColumns];
    int *h_C = new int[numARows * numBColumns];

    // Initialize matrices A and B with random values between 0 and 9
    srand(time(NULL));
    for (int i = 0; i < numARows * numAColumns; i++) {
        h_A[i] = rand() % 10;
    }
    for (int i = 0; i < numBRows * numBColumns; i++) {
        h_B[i] = rand() % 10;
    }

    // Allocate space for matrices on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy matrices from the host to the device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(16, 16);
    dim3 dimGrid((numBColumns + dimBlock.x - 1) / dimBlock.x, (numARows + dimBlock.y - 1) / dimBlock.y);

    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                matrixMultiplyCUDA<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
                break;
        }
        timer.stop();
        time_taken = timer.getElapsedTime();
        sum += time_taken;
        i++;
    } while(i < 100);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    printf("total time (nanoseconds): %f\n", sum);
    printf("average time (nanoseconds): %f\n", (double)sum / i);
}