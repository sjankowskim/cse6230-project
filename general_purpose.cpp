#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../headers/chatgpt.hpp"
#include "../headers/utils.hpp"

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

int main(int argc, char *argv[]) {
    int type;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) {
            int num = atoi(argv[i + 1]);
            if (num > 2 || num < 0) {
                printf("okay, smartass.\n");
                return 1;
            }
            type = num;
            i++;

            switch (type) {
                case 0:
                    printf("Using GPT-3!\n");
                    break;
                case 1:
                    printf("Using library call!\n");
                    break;
                case 2:
                    printf("Using GPT-4!\n");
                    break;
            }
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("./test_code <flags>\n"
                    "\t-t [num]     : Determines what type of output to use (0: GPT-3, 1: library, 2: GPT-4)\n");
            return 0;
        } else {
            printf("./test_code <flags>\n"
                    "\t-t [num]     : Determines what type of output to use (0: GPT-3, 1: library, 2: GPT-4)\n");
            return 0;
        }
    }

    double angle = 0.0;
    uint64_t time_taken;
    double sum = 0;
    int i = 0;

    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        switch (type) {
            case 0:
                gpt_cosine(angle);
                break;
            case 1:
                cos(angle);
                break;
            case 2:
                // TODO: ChatGPT-4
                break;
        }
        timer.stop();
        time_taken = timer.getElapsedTime();
        sum += time_taken;
        i++;
        angle += 0.0000001;
    } while(angle <= PI);

    printf("total time (nanoseconds): %f\n", sum);
    printf("average time (nanoseconds): %f\n", (double)sum / i);
}