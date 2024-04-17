#include <cuda.h>
#include "../../../utils.hpp"
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE       (256)
#define NUM_TRIALS      (1000)
#define N             (100000)

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY AN LLM!           |
 *-------------------------------*/

__global__ void gpt_findTargetValue(const int* array, int size, int target, int** result) {
    __shared__ int* minPtr; // Shared memory variable to store the minimum pointer to the target value

    // Initialize minPtr to nullptr
    if (threadIdx.x == 0) {
        minPtr = nullptr;
    }
    __syncthreads();

    // Search for the target value in parallel
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (array[i] == target) {
            int* myPtr = const_cast<int*>(&array[i]); // Pointer to the target value found by this thread

            // Update minPtr using atomic operation
            atomicMin(reinterpret_cast<unsigned long long*>(&minPtr), reinterpret_cast<unsigned long long>(myPtr));
        }
    }
    __syncthreads();

    // Output the pointer to the first instance of the target value
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = minPtr;
    }
}

__global__ void my_findTarget(const int* array, int size, int target, int** result) {
    for (int i = 0; i < size; i++) {
        if (array[i] == target) {
            *result = (int*)&array[i];
            return;
        }
    }
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Timer<std::milli> timer;
    bool assertion = true;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // TODO: Setup thrust variables
    thrust::device_vector<int> thrust_in(N);
    thrust::device_vector<int>::iterator thrust_result;

    // TODO: Setup initial variables
    int* in;
    int** d_result;
    int* h_result;

    // TODO: cudaMalloc as needed
    cudaMalloc(&in, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int*));

    // TODO: Setup initial variables and cudaMemcpy as needed.
    int temp[N];
    for (int k = 0; k < N; k++) {
        temp[k] = k;
        thrust_in[k] = k;
    }
    cudaMemcpy(in, temp, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < 5; i++) {
        double sum = 0;

        switch (i) {
            case LIBRARY:
                printf("Testing library call!\n");
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
            switch (i) {
                case LIBRARY:
                    timer.start();
                    thrust_result = thrust::find(thrust_in.begin(), thrust_in.end(), 69);
                    timer.stop();
                    break;
                case GPT3:
                    cudaEventRecord(start, 0);
                    gpt_findTargetValue<<<num_blocks, BLOCK_SIZE>>>(in, N, 69, d_result);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    break;
                case GPT4:
                    // TODO: GPT4
                    cudaEventRecord(start, 0);
                    my_findTarget<<<1, 1>>>(in, N, 69, d_result);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    break;
                case COPILOT:
                    break;
                case GEMINI:
                    break;
            }
            
            if (j != 0) {
                if (i == 0){
                    sum += timer.getElapsedTime();
                } else {
                    float elapsedTime;
                    cudaEventElapsedTime(&elapsedTime, start, stop);
                    sum += elapsedTime;
                }
            }

            // TODO: Verify results with library
            if (i != 0) {
                int* h_temp_result;
                h_result = thrust::find(thrust::device, in, in + N, 69);
                cudaMemcpy(&h_temp_result, d_result, sizeof(int*), cudaMemcpyDeviceToHost);
                assertion = (h_result == h_temp_result);
                if (!assertion) {
                    printf("\tintended_result: %p, actual result: %p\n", h_result, h_temp_result);
                    break;
                }
            }
        }

        if (!assertion) {
            printf("\tIncorrect output! Continuing...\n");
            continue;
        }
        printf("\ttotal avg time (milliseconds): %f\n", sum / (NUM_TRIALS - 1));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // TODO: Free as needed
    cudaFree(d_result);
    cudaFree(in);
}