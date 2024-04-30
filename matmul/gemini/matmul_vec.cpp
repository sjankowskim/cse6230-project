#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../utils.hpp"

#include <iostream>
#include <vector>
#include <immintrin.h>
#include <random>

using namespace std;
/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

const int MATRIX_SIZE = 100;
const int ALIGNMENT = 32;  // Alignment for AVX instructions (multiple of cache line size)

typedef __m256i vec_t;  // Define vector type for AVX intrinsics

void fillRandomMatrix(int matrix[][MATRIX_SIZE]) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0, 9);

  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; j++) {
      matrix[i][j] = distribution(generator);
    }
  }
}

void multiplyMatricesAVX(const int* aligned_A, const int* aligned_B, int* result) {
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; j += 8) {  // Process 8 elements per loop
      vec_t vec_sum = _mm256_setzero_si256();
      for (int k = 0; k < MATRIX_SIZE; k += 4) {
        // Load vectors from matrices
        vec_t vec_a = _mm256_load_align_epi32((const int*)(aligned_A + i * MATRIX_SIZE + k), ALIGNMENT);
        vec_t vec_b = _mm256_load_align_epi32((const int*)(aligned_B + k * MATRIX_SIZE + j), ALIGNMENT);

        // Perform element-wise multiplication and accumulate
        vec_sum = _mm256_add_epi32(vec_sum, _mm256_mul_epu32(vec_a, vec_b));
      }

      // Store the calculated results (cast back to int)
      _mm256_store_u((int*)(result + i * MATRIX_SIZE + j), vec_sum);
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

  int *matrixA_h = (int*)aligned_alloc(ALIGNMENT, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
  int *matrixB_h = (int*)aligned_alloc(ALIGNMENT, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
  int *result_h = (int*)aligned_alloc(ALIGNMENT, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

  // Fill matrices with random values
  fillRandomMatrix(matrixA_h);
  fillRandomMatrix(matrixB_h);

    uint64_t time_taken;
    double sum = 0;
    int i = 0;

    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                multiplyMatricesAVX(matrixA_h, matrixB_h, result_h); 
            break;
        }
        timer.stop();
        time_taken = timer.getElapsedTime();
        sum += time_taken;
        i++;
    } while(i < 100);

      free(matrixA_h);
  free(matrixB_h);
  free(result_h);

    printf("total time (nanoseconds): %f\n", sum);
    printf("average time (nanoseconds): %f\n", (double)sum / i);
}