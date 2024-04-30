#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to perform matrix multiplication
void matrixMultiply(const float* A, const float* B, float* C, int numRows, int N) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int rank, size, N = 100;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numRows = N / size;  // Number of rows handled by each process
    if (N % size != 0 && rank == 0) {
        std::cerr << "Matrix size not divisible evenly by number of processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float *A, *B, *C, *subA, *subC;
    B = new float[N * N];      // B is entirely used by all processes
    subA = new float[numRows * N];  // Portion of A
    subC = new float[numRows * N];  // Portion of C

    // Initialize and distribute matrices
    if (rank == 0) {
        A = new float[N * N];
        C = new float[N * N];
        srand(static_cast<unsigned>(time(nullptr)));
        for (int i = 0; i < N * N; i++) {
            A[i] = static_cast<float>(rand() % 10);
            B[i] = static_cast<float>(rand() % 10);  // Initialize B in rank 0 and broadcast later
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting timing
    double start_time = MPI_Wtime();

    // Broadcast B to all processes
    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter A to all processes
    MPI_Scatter(A, numRows * N, MPI_FLOAT, subA, numRows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Matrix multiplication
    matrixMultiply(subA, B, subC, numRows, N);

    // Gather results back to the root process
    MPI_Gather(subC, numRows * N, MPI_FLOAT, C, numRows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Total time taken for matrix multiplication: " << (end_time - start_time) << " seconds." << std::endl;
        // Optionally print a small part of the matrix to check results
        for (int i = 0; i < std::min(10, N); i++) {
            for (int j = 0; j < std::min(10, N); j++) {
                std::cout << C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        delete[] A;
        delete[] C;
    }

    delete[] B;
    delete[] subA;
    delete[] subC;

    MPI_Finalize();
    return 0;
}