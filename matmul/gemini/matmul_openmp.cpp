#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../utils.hpp"

#include <iostream>
#include <omp.h>

/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

const int MATRIX_SIZE = 100;

void fillRandomMatrix(int matrix[][MATRIX_SIZE]) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0, 9);

  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      matrix[i][j] = distribution(generator);
    }
  }
}

void multiplyMatricesOpenMP(const int matrixA[][MATRIX_SIZE], const int matrixB[][MATRIX_SIZE], int result[][MATRIX_SIZE]) {
  #pragma omp parallel for collapse(2)  // Parallelize outer and inner loops
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      result[i][j] = 0;
      for (int k = 0; k < MATRIX_SIZE; ++k) {
        result[i][j] += matrixA[i][k] * matrixB[k][j];
      }
    }
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

  int matrixA[MATRIX_SIZE][MATRIX_SIZE];
  int matrixB[MATRIX_SIZE][MATRIX_SIZE];
  int result[MATRIX_SIZE][MATRIX_SIZE];

  // Fill matrices with random values
  fillRandomMatrix(matrixA);
  fillRandomMatrix(matrixB);


    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                multiplyMatricesOpenMP(matrixA, matrixB, result);
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