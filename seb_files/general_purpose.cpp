#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "utils.hpp"

/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

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

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/

#define PI (3.14)

int main() {
    Timer<std::nano> timer;
    double angle;
    double sum;
    double result;
    int count;

    for (int i = 0; i < 2; i++) {
        count = 0;
        angle = 0.0;
        sum = 0;

        do {
            timer.start();
            switch (i) {
                case 0:
                    result = gpt_cosine(angle);
                    break;
                case 1:
                    cos(angle);
                    break;
                case 2:
                    // TODO: ChatGPT-4
                    break;
            }
            timer.stop();

            // if (i == 0) {
            //     printf("chatgpt: %f\n", result);
            //     printf("library: %f\n", cos(angle));
            //     assert(result == cos(angle));
            // }
            sum += timer.getElapsedTime();
            angle += 0.000001;
            count++;
        } while(angle <= PI);

        switch (i) {
            case 0:
                printf("Testing GPT-3.5!\n");
                break;
            case 1:
                printf("Testing library call!\n");
                break;
            case 2:
                // TODO: ChatGPT-4
                break;
        }
        printf("total time (nanoseconds): %f\n", sum);
        printf("average time (nanoseconds): %f\n", (double)sum / count);
    }
}