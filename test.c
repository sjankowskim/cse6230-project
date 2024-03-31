#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include "chatgpt.h"

#define PI (3.14)

extern inline __attribute((always_inline))
uint64_t rdtscp(void) {
    uint64_t cycles;
    asm volatile ("rdtscp\n"
                "shl $32, %%rdx\n"
                "or %%rdx, %%rax\n"
                : "=a" (cycles));
    return cycles;
}

int main() {
    double angle = 0.0;
    uint64_t curr_time;
    uint64_t time_taken;
    double result;

    uint64_t sum = 0;
    int i = 0;

    do {
        curr_time = rdtscp();
        // result = cosine(angle);
        result = cos(angle);
        time_taken = rdtscp() - curr_time;
        // printf("cosine output: %f\n", result);
        // printf("cycles taken: %ld\n", time_taken);
        sum += time_taken;
        i++;
        angle += 0.0000001;
    } while(angle <= PI);

    printf("average cycles: %f\n", (double)sum / i);
}