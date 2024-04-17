#include <cuda.h>
#include "../../../utils.hpp"

__global__ void prefix_sum(int* A, int* B, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_sum = 0; // Local thread sum

  // Calculate prefix sum for the first element efficiently (no reduction)
  if (i == 0) {
    thread_sum = A[0];
  } else {
    // Reduction within a warp to efficiently calculate prefix sum for remaining elements
    __shared__ int shared_sum[32];  // Shared memory for warp-level reduction
    int offset = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    if (lane_idx == 0) {
      shared_sum[lane_idx] = (i < n) ? A[i] : 0;
    }
    __syncthreads();

    for (int stride = 1; stride <= 16; stride *= 2) {
      if (lane_idx < stride) {
        shared_sum[lane_idx] += shared_sum[lane_idx + stride];
      }
      __syncthreads();
    }

    // Final accumulation considering warp offset
    thread_sum = (offset * 32 + lane_idx < n) ? shared_sum[lane_idx] : 0;
  }

  // Add prefix sum from previous block (scan approach)
  if (i > 0) {
    thread_sum += B[i - 1];
  }

  // Store the result in the output array
  if (i < n) {
    B[i] = thread_sum;
  }
}

int main() {
  // Host memory allocation (replace n with your array size)
  int n = 4096;
  int A[n];
  int B[n]; // Output array for prefix sum

  // Initialize array (replace with your initialization logic)
  for (int i = 0; i < n; i++) {
    A[i] = i + 1; // Example initialization
  }

  // Device memory allocation
  int *A_d, *B_d;
  cudaMalloc(&A_d, n * sizeof(int));
  cudaMalloc(&B_d, n * sizeof(int));

  // Copy data from host to device
  cudaMemcpy(A_d, A, n * sizeof(int), cudaMemcpyHostToDevice);

  // Kernel launch with appropriate grid size
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  prefix_sum<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, n);

  // Copy result from device to host (optional)
  cudaMemcpy(B, B_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  // Print the prefix sum (optional)
  for (int i = 0; i < n; i++) {
    printf("Prefix sum at %d: %d\n", i, B[i]);
  }

  // Free device memory
  cudaFree(A_d);
  cudaFree(B_d);

  return 0;
}