#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "utils.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <random>

/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/
const int MATRIX_SIZE = 100;

__global__ void multiplyMatrices(const int* matrixA, const int* matrixB, int* result) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
    int sum = 0;
    for (int k = 0; k < MATRIX_SIZE; ++k) {
      sum += matrixA[row * MATRIX_SIZE + k] * matrixB[k * MATRIX_SIZE + col];
    }
    result[row * MATRIX_SIZE + col] = sum;
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

  int *matrixA_h, *matrixB_h, *result_h;
  matrixA_h = new int[MATRIX_SIZE * MATRIX_SIZE];
  matrixB_h = new int[MATRIX_SIZE * MATRIX_SIZE];
  result_h = new int[MATRIX_SIZE * MATRIX_SIZE];

  // Seed random number generator
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0, 9);

  // Fill matrices with random values on host
  for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
    matrixA_h[i] = distribution(generator);
    matrixB_h[i] = distribution(generator);
  }

  // Allocate memory on device for matrices
  int *matrixA_d, *matrixB_d, *result_d;
  cudaMalloc(&matrixA_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
  cudaMalloc(&matrixB_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
  cudaMalloc(&result_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

  // Copy matrices from host to device
  cudaMemcpy(matrixA_d, matrixA_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(matrixB_d, matrixB_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

  // Define thread block size (adjust based on your GPU)
  int threadsPerBlock = 16;
  int blocksPerGrid = (MATRIX_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                multiplyMatrices<<<blocksPerGrid, threadsPerBlock>>>(matrixA_d, matrixB_d, result_d);
                break;
        }
        timer.stop();
        time_taken = timer.getElapsedTime();
        sum += time_taken;
        i++;
    } while(i < 100);

    cudaMemcpy(result_h, result_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(matrixA_d);
  cudaFree(matrixB_d);
  cudaFree(result_d);

  // Deallocate memory on host
  delete[] matrixA_h;
  delete[] matrixB_h;
  delete[] result_h;

    printf("total time (nanoseconds): %f\n", sum);
    printf("average time (nanoseconds): %f\n", (double)sum / i);
}