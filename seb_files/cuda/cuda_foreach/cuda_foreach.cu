#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/for_each.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

__global__ void gpt3_squareKernel(int *input, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        input[tid] *= input[tid];
    }
}

// CUDA kernel function to square elements
__global__ void gpt4_squareElements(int *d_in, int *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

__global__ void copilot_square(int *d_out, int *d_in, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        float f = d_in[idx];
        d_out[idx] = f * f;
    }
}

__global__ void gemini_square_array(int* A, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check for threads exceeding valid array bounds
  if (i < n) {
    A[i] *= A[i]; // Square the element in-place
  }
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

struct square_ref_t
{
  __device__ void operator()(int& i)
  {
    i *= i;
  }
};

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Timer<std::nano> timer;
    bool assertion = true;
    std::chrono::duration<double, std::nano> sum(0);
    square_ref_t op{};

    // TODO: Setup initial variables
    int *in;
    thrust::device_vector<int> d_vec;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));

    // TODO: Setup CUB stuff as needed
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceFor::ForEach(
        d_temp_storage, temp_storage_bytes, d_vec.begin(), d_vec.end(), op);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    for (int i = 0; i < 6; i++) {
        sum = std::chrono::duration<double, std::nano>(0);

        switch (i) {
            case CUB:
                printf("Testing CUB!\n");
                break;
            case THRUST:
                printf("Testing Thrust!\n");
                break;
            case GPT3:
                printf("Testing GPT-3.5!\n");
                break;
            case GPT4:
                printf("Testing GPT-4!\n");
                break;
            case COPILOT:
                printf("Testing Copilot!\n");
                break;
            case GEMINI:
                printf("Testing Gemini!\n");
                break;
        }

        for (int j = 0; j < NUM_TRIALS; j++) {

            // TODO: Setup initial variables and cudaMemcpy as needed.
            srand(j * i);
            int temp[N];
            for (int k = 0; k < N; k++) {
                temp[k] = rand() % 1000;
            }
            cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);
            d_vec = thrust::device_vector<int>(temp, temp + N);

            switch (i) {
                case CUB:
                    timer.start();
                    cub::DeviceFor::ForEach(
                        d_temp_storage, temp_storage_bytes, d_vec.begin(), d_vec.end(), op);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case THRUST:
                    timer.start();
                    thrust::for_each(thrust::device, in, in + N, square_ref_t());
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT3:
                    timer.start();
                    gpt3_squareKernel<<<num_blocks, BLOCK_SIZE>>>(in, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GPT4:
                    timer.start();
                    gpt4_squareElements<<<num_blocks, BLOCK_SIZE>>>(in, in, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case COPILOT:
                    timer.start();
                    copilot_square<<<num_blocks, BLOCK_SIZE>>>(in, in, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
                case GEMINI:
                    timer.start();
                    gemini_square_array<<<num_blocks, BLOCK_SIZE>>>(in, N);
                    cudaDeviceSynchronize();
                    timer.stop();
                    break;
            }
            
            if (j != 0) {
                sum += timer.getElapsedTimeChrono();
            }

            // TODO: Verify results with library
            if (i != 0 && i != 1) {
                thrust::device_vector<int> temp_d_vec(temp, temp + N);
                cub::DeviceFor::ForEach(
                    d_temp_storage, temp_storage_bytes, temp_d_vec.begin(), temp_d_vec.end(), op);
                assertion = thrust::equal(thrust::device, in, in + N, temp_d_vec.data());
                if (!assertion) {
                    break;
                }
            }
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (nanoseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    // TODO: Free as needed
    cudaFree(in);
    cudaFree(d_temp_storage);
}