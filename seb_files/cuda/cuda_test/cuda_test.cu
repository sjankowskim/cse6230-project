#include <iostream>
#include <cuda_runtime.h>

__global__ void sumKernel(int *input, int size, int *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int sum = 0;

    // Calculate sum of elements by each thread
    while (tid < size) {
        sum += input[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Reduce sum across threads in a block
    __shared__ int sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, sdata[0]);
    }
}

void calculateSum(int *input, int size, int *result) {
    int *d_input, *d_result;

    cudaMalloc((void **)&d_input, size * sizeof(int));
    cudaMalloc((void **)&d_result, sizeof(int));

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    sumKernel<<<gridSize, blockSize>>>(d_input, size, d_result);

    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);
}

int main() {
    const int size = 1000000;
    int *array = new int[size];
    for (int i = 0; i < size; ++i) {
        array[i] = i; // Example array initialization
    }

    int sum = 0;
    calculateSum(array, size, &sum);

    std::cout << "Sum of array elements: " << sum << std::endl;

    delete[] array;

    return 0;
}