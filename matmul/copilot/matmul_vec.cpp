#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../utils.hpp"

#include <iostream>
#include <vector>
#include <immintrin.h>
/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/
// Matrix dimensions
const int N = 100;

// Generate random values in the range [0, 9]
float getRandomValue() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 9.0f);
    return dist(gen);
}

void matrixMultiplyAVX2(float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; j += 8) {
            __m256 rowA = _mm256_loadu_ps(&A[i * N]); // Load row from A
            __m256 rowB = _mm256_loadu_ps(&B[j]);     // Load row from B

            // Multiply and accumulate
            __m256 result = _mm256_mul_ps(rowA, rowB);
            for (int k = 1; k < 8; ++k) {
                rowA = _mm256_loadu_ps(&A[i * N + k]);
                rowB = _mm256_loadu_ps(&B[j + k]);
                result = _mm256_fmadd_ps(rowA, rowB, result);
            }

            // Store the result
            _mm256_storeu_ps(&C[i * N + j], result);
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
    float A[N * N], B[N * N], C[N * N];

    // Initialize matrices with random values
    for (int i = 0; i < N * N; ++i) {
        A[i] = getRandomValue();
        B[i] = getRandomValue();
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
                // Multiply matrices using AVX2
    matrixMultiplyAVX2(A, B, C);
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