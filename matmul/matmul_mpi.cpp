#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../../utils.hpp"

#include <iostream>
#include <vector>
#include <mpi.h>

/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/
using namespace std;

vector<vector<int>> matrix_multiply(const vector<vector<int>>& A, const vector<vector<int>>& B, int rows_A, int cols_A, int cols_B, int rank, int size) {
    // Initialize result matrix with 0s
    vector<vector<int>> result(rows_A, vector<int>(cols_B, 0));

    // Create buffers to hold entire matrix data in a contiguous block of memory
    vector<int> buffer_A(rows_A * cols_A);
    vector<int> buffer_B(cols_A * cols_B);

    // Copy matrix A into buffer_A
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            buffer_A[i * cols_A + j] = A[i][j];
        }
    }

    // Copy matrix B into buffer_B
    if (rank == 0) {
        for (int i = 0; i < cols_A; ++i) {
            for (int j = 0; j < cols_B; ++j) {
                buffer_B[i * cols_B + j] = B[i][j];
            }
        }
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(buffer_B.data(), cols_A * cols_B, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter matrix A among processes
    vector<int> local_A(rows_A * cols_A / size);
    MPI_Scatter(buffer_A.data(), rows_A * cols_A / size, MPI_INT, local_A.data(), rows_A * cols_A / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication locally
    for (int i = 0; i < rows_A / size; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            for (int k = 0; k < cols_A; ++k) {
                result[i + rank * rows_A / size][j] += local_A[i * cols_A + k] * buffer_B[k * cols_B + j];
            }
        }
    }

    // Gather results
    MPI_Gather(result.data()[0], rows_A * cols_B / size, MPI_INT, result.data()[0], rows_A * cols_B / size, MPI_INT, 0, MPI_COMM_WORLD);

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

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        cerr << "This program must be run with at least 2 processes" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
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

    vector<vector<int>> result;

    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                if (rank == 0) {
                    result = matrix_multiply(A, B, rows_A, cols_A, cols_B, rank, size);
                cout << "Result of multiplication:" << endl;
                } else {
                    matrix_multiply(A, B, rows_A, cols_A, cols_B, rank, size);
                }

                MPI_Finalize();
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