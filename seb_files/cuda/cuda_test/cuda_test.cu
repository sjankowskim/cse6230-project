#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/gather.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

__global__ void filter_and_count(int* A, int* flag, int n, int* B, int* num_active) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int write_idx = 0; // Index for writing to output array (atomic)

  // Check for threads exceeding valid array bounds
  if (i < n) {
    if (flag[i] == 1) {
      // Use atomic operations for thread-safe writing to output
      B[atomicAdd(&write_idx, 1) - 1] = A[i];
      atomicAdd(num_active, 1);
    }
  }
}

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

int main() {
  // Host memory allocation (replace n with your array size)
  int n = N;
  int A[n];
  int flag[n]; // Flag array with 1s and 0s
  int B[n]; // Output array to store filtered elements
  int num_active = 0; // Counter for active flags (atomic)

  // Initialize arrays (replace with your initialization logic)
  for (int i = 0; i < n; i++) {
    A[i] = rand(); // Example initialization for input array
    flag[i] = rand() % 2; // Example initialization for flag array (alternating 1s and 0s)
  }

  // Device memory allocation
  int *A_d, *flag_d, *B_d;
  cudaMalloc(&A_d, n * sizeof(int));
  cudaMalloc(&flag_d, n * sizeof(int));
  cudaMalloc(&B_d, n * sizeof(int));

  // Copy data from host to device
  cudaMemcpy(A_d, A, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(flag_d, flag, n * sizeof(int), cudaMemcpyHostToDevice);

  // Kernel launch with appropriate grid size
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  Timer<std::nano> timer;
  std::chrono::duration<double, std::nano> sum(0);
  
  for (int i = 0; i < NUM_TRIALS; i++) {
    timer.start();
    filter_and_count<<<blocksPerGrid, threadsPerBlock>>>(A_d, flag_d, n, B_d, &num_active);
    cudaDeviceSynchronize();
    timer.stop();
    sum += timer.getElapsedTimeChrono();
  }

  printf("sum: %f\n", sum / NUM_TRIALS);
  // Allocate space to store the number of active flags on host
  int num_active_host;

  // Copy result from device to host
  cudaMemcpy(&num_active_host, &num_active, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(B, B_d, n * sizeof(int), cudaMemcpyDeviceToHost); // Optional (copy filtered elements if needed)

  // Print results (optional)
  printf("Number of active flags: %d\n", num_active_host);

  // Free device memory
  cudaFree(A_d);
  cudaFree(flag_d);
  cudaFree(B_d);

  return 0;
}