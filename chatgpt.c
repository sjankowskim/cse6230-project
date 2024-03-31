/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

// Function to calculate cosine using Taylor series expansion
double cosine(double x) {
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

// 1) General algorithms (small)
// 2) Vectorization
// 3) CUDA 
// 4) OpenMP

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/