/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

// Function to calculate cosine using Taylor series expansion
double gpt_cosine(double x) {
    double result = 1.0;
    double term = 1.0;
    int sign = -1;

    // Iterate to add more terms to the series
    for (int i = 2; i <= 20; i += 2) {  // Using 20 terms for approximation
        term = term * x * x / (i * (i - 1));
        result += sign * term;
        sign *= -1;
    }

    return result;
}

__global__ void gpt_inclusiveScan(int* input, int* output, int n) {
    __shared__ int temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Copy input to shared memory
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0;
    }

    __syncthreads();

    // Reduction phase
    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE * 2) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Post-reduction phase
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE * 2) {
            temp[index + stride] += temp[index];
        }
    }

    __syncthreads();

    // Write result to output
    if (idx < n) {
        output[idx] = temp[tid];
    }
}

// 1) General algorithms (small)
// 2) Vectorization
// 3) CUDA 
// 4) OpenMP

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/