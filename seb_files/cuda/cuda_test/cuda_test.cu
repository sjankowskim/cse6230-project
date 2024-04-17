#include <iostream>
#include <cuda_runtime.h>

__global__ void squareKernel(float *input, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        input[tid] *= input[tid];
    }
}

void squareArray(float *input, int size) {
    float *d_input;
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    squareKernel<<<gridSize, blockSize>>>(d_input, size);

    cudaMemcpy(input, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
}

int main() {
    const int size = 1000000;
    float *array = new float[size];
    for (int i = 0; i < size; ++i) {
        array[i] = i; // Example array initialization
    }

    squareArray(array, size);

    // Print the squared array
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    delete[] array;
    return 0;
}
