#include <algorithm>
#include <execution>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <ratio>
#include <vector>

#include "../utils.hpp"

constexpr int DEFAULTSIZE = 10'000'000;
constexpr int DEFAULTSEED = 1;
constexpr int DEFAULTREPS = 100;
constexpr int MAX_DEPTH = 8;

/*
This file is meant to be copied and used as a framework
for your other files. Hope it helps!
*/

/*-------------------------------*
 | CODE WRITTEN IN THIS SECITON  |
 | WAS DONE BY CHATGPT!          |
 *-------------------------------*/

int reduceVectorParallel(const std::vector<int>& vec) {
    int sum = 0;
    size_t n = vec.size();

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        sum += vec[i];
    }

    return sum;
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/


int main(int argc, char** argv)
{
    size_t size = find_int_arg(argc, argv, "-n", DEFAULTSIZE);
    int seed = find_int_arg(argc, argv, "-s", DEFAULTSEED);
    auto policy = std::execution::par;

    Timer timer;

    std::vector<int> libraryArr(size);
    std::vector<int> gptArr(size);

    // Additional vectors for verification
    std::vector<int> libraryReductions(DEFAULTREPS);
    std::vector<int> gptReductions(DEFAULTREPS);

    auto stlLambda = [&] () {
        // Time function
        timer.start();
        int red = std::reduce(policy, libraryArr.begin(), libraryArr.end());
        timer.stop();

        // Insert for verification
        libraryReductions.push_back(red);

        return timer.getElapsedTime();
    };

    auto gptLambda = [&] () {
        // Time function
        timer.start();
        int red = reduceVectorParallel(gptArr);
        timer.stop();

        // Insert for verification
        gptReductions.push_back(red);

        return timer.getElapsedTime();
    };

    auto averageLibraryTime = runTests(libraryArr, DEFAULTREPS, stlLambda, size, seed);
    printStats("Library Average: ", averageLibraryTime, "ns");
    auto averageGPTTime = runTests(gptArr, DEFAULTREPS, gptLambda, size, seed);
    printStats("ChatGPT Average: ", averageGPTTime, "ns");

    return 0;
}