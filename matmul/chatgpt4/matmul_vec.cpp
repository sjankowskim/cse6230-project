#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../utils.hpp"

#include <iostream>
#include <vector>
#include <immintrin.h>

using namespace std;

const int VECTOR_SIZE = 8;
/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

void matrixMultiplyAVX2(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum = _mm256_setzero_ps();  // Initialize sum vector to 0
            for (int k = 0; k < N; k += 8) {
                __m256 a = _mm256_loadu_ps(&A[i * N + k]);  // Load 8 elements from A
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);  // Load 8 elements from B
                sum = _mm256_fmadd_ps(a, b, sum);           // Fused multiply-add
            }
            sum = _mm256_hadd_ps(sum, sum);  // Horizontal add to reduce the vector
            sum = _mm256_hadd_ps(sum, sum);
            float buffer[8];
            _mm256_storeu_ps(buffer, sum);
            C[i * N + j] = buffer[0] + buffer[4];  // Store the result in C
        }
    }
}
/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

#define PI (3.14)

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

    // Seed the random number generator
    srand(time(nullptr));

    const int N = 100;  // Matrix dimension
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // Initialize matrices A and B with random values between 0 and 9
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand() % 10);
        B[i] = static_cast<float>(rand() % 10);
    }

    uint64_t time_taken;
    double sum = 0;
    int i = 0;

    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                matrixMultiplyAVX2(A, B, C, N);   
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