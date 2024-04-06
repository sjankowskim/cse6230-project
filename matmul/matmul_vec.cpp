#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../../utils.hpp"

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
vector<vector<int>> matrix_multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int cols_B = B[0].size();

    // Result matrix initialized with 0s
    vector<vector<int>> result(rows_A, vector<int>(cols_B, 0));

    // Transpose matrix B for better memory access pattern
    vector<vector<int>> transposed_B(cols_A, vector<int>(cols_B));
    for (int i = 0; i < cols_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            transposed_B[i][j] = B[j][i];
        }
    }

    uint64_t time_taken;
    Timer<std::nano> timer;
    timer.start();
    // Matrix multiplication
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            __m256i sum = _mm256_setzero_si256(); // Initialize sum vector to all zeros

            for (int k = 0; k < cols_A; k += VECTOR_SIZE) {
                __m256i a_vec = _mm256_loadu_si256((__m256i*)&A[i][k]); // Load vector of elements from A
                __m256i b_vec = _mm256_loadu_si256((__m256i*)&transposed_B[j][k]); // Load vector of elements from B

                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(a_vec, b_vec)); // Multiply and add vectors
            }

            // Horizontal addition of elements in the sum vector
            int sum_array[VECTOR_SIZE];
            _mm256_storeu_si256((__m256i*)sum_array, sum);
            for (int l = 0; l < VECTOR_SIZE; ++l) {
                result[i][j] += sum_array[l];
            }
        }
    }
    timer.stop();
    time_taken = timer.getElapsedTime();

    return time_taken;
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

    uint64_t time_taken;
    double sum = 0;
    int i = 0;

    Timer<std::nano> timer;
    // comment
    do {
        switch (type) {
            case 0:
                time_taken = matrix_multiply(A, B);
                break;
            case 1:
                break;
            case 2:
                // TODO: ChatGPT-4
                break;
        }
        sum += time_taken;
        i++;
    } while(i < 100);

    printf("total time (nanoseconds): %f\n", sum);
    printf("average time (nanoseconds): %f\n", (double)sum / i);
}