#include <iostream>
#include <mpi.h>
#include <random>

const int MATRIX_SIZE = 100;

void fillRandomMatrix(int matrix[][MATRIX_SIZE], int rank, int size) {
  std::random_device rd;
  std::mt19937 generator(rank + rd());  // Seed generator with rank for different values on each process
  std::uniform_int_distribution<int> distribution(0, 9);

  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      matrix[i][j] = distribution(generator);
    }
  }
}

void multiplyMatricesMPI(const int matrixA[][MATRIX_SIZE], const int matrixB[][MATRIX_SIZE], int result[][MATRIX_SIZE],
                          int rank, int size) {
  // Scatter rows of matrixA to all processes (block distribution)
  MPI_Scatter(matrixA, MATRIX_SIZE * sizeof(int) / size, MPI_INT, MPI_INT, 0, MPI_COMM_WORLD);

  // Broadcast matrixB to all processes
  MPI_Bcast(matrixB, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);

  // Local portion of the result matrix
  int local_result[MATRIX_SIZE][MATRIX_SIZE];

  // Perform multiplication for each row received (local portion of A)
  for (int i = 0; i < MATRIX_SIZE / size; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      local_result[i][j] = 0;
      for (int k = 0; k < MATRIX_SIZE; ++k) {
        local_result[i][j] += matrixA[i * size + rank][k] * matrixB[k][j];
      }
    }
  }

  // Gather local results from all processes into the final result on process 0
  MPI_Gather(local_result, MATRIX_SIZE * sizeof(int) / size, MPI_INT, result, MATRIX_SIZE * sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int matrixA[MATRIX_SIZE][MATRIX_SIZE];
  int matrixB[MATRIX_SIZE][MATRIX_SIZE];
  int result[MATRIX_SIZE][MATRIX_SIZE];

  // Fill matrices with random values (different values on each process)
  fillRandomMatrix(matrixA, rank, size);
  fillRandomMatrix(matrixB, rank, size);  // Can be filled with same values on all processes

  // Multiply matrices using MPI
  multiplyMatricesMPI(matrixA, matrixB, result, rank, size);

  // Only process 0 prints the result (optional)
  if (rank == 0) {
    // Print the result (can be slow for large matrices)
    // ... (same logic as before)
  }

  MPI_Finalize();

  return 0;
}