#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "../headers/chatgpt.h"
#include "../headers/utils.h"

#define PI (3.14)

int main() {
    double angle = 0.0;
    uint64_t time_taken;
    double sum = 0;
    int i = 0;

    Timer<std::nano> timer;
    // comment
    do {
        timer.start();
        // result = cosine(angle);
        cos(angle);
        timer.stop();
        time_taken = timer.getElapsedTime();
        // printf("cosine output: %f\n", result);
        // printf("cycles taken: %ld\n", time_taken);
        sum += time_taken;
        i++;
        angle += 0.0000001;
    } while(angle <= PI);
    printf("total time: %f\n", sum);
    printf("average cycles: %f\n", (double)sum / i);
}