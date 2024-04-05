#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../../utils.hpp"

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

// CUDA kernel to perform matrix multiplication
__global__ void matrix_multiply_kernel(int* A, int* B, int* C, int rows_A, int cols_A, int cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_A && col < cols_B) {
        int sum = 0;
        for (int k = 0; k < cols_A; ++k) {
            sum += A[row * cols_A + k] * B[k * cols_B + col];
        }
        C[row * cols_B + col] = sum;
    }
}

// Host function to perform matrix multiplication
vector<vector<int>> matrix_multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int cols_B = B[0].size();

    // Result matrix initialized with 0s
    vector<vector<int>> result(rows_A, vector<int>(cols_B, 0));

    // Device memory allocation
    int* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rows_A * cols_A * sizeof(int));
    cudaMalloc(&d_B, cols_A * cols_B * sizeof(int));
    cudaMalloc(&d_C, rows_A * cols_B * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A.data(), rows_A * cols_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), cols_A * cols_B * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((cols_B + blockSize.x - 1) / blockSize.x, (rows_A + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrix_multiply_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_A, cols_A, cols_B);

    // Copy result matrix C from device to host
    cudaMemcpy(result.data(), d_C, rows_A * cols_B * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
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

    // Seed the random number generator
    srand(time(nullptr));

    // Define matrix dimensions
    int rows_A = 100;
    int cols_A = 100;
    int cols_B = 100;

    // Populate matrix A with random values
    vector<vector<int>> A(rows_A, vector<int>(cols_A, 0));
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            A[i][j] = rand() % 10; // Generate random numbers between 0 and 9
        }
    }

    // Populate matrix B with random values
    vector<vector<int>> B(cols_A, vector<int>(cols_B, 0));
    for (int i = 0; i < cols_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            B[i][j] = rand() % 10; // Generate random numbers between 0 and 9
        }
    }


    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                matrix_multiply(A,B);
                break;
        }
        timer.stop();
        time_taken = timer.getElapsedTime();
        sum += time_taken;
        i++;
    } while(i < 100);

    printf("total time (nanoseconds): %f\n", sum);
    printf("average time (nanoseconds): %f\n", (double)sum / i);
}