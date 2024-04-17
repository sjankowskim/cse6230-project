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

// Template function to count elements in a vector equal to a specific value using OpenMP
template<typename T>
size_t countElementsParallel(const std::vector<T>& vec, const T& value) {
    size_t count = 0;
    #pragma omp parallel for reduction(+:count) // Parallelize the loop with reduction on count
    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] == value) {
            count++;
        }
    }
    return count;
}

/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/


int main(int argc, char** argv)
{
    size_t size = find_int_arg(argc, argv, "-n", DEFAULTSIZE);
    int seed = find_int_arg(argc, argv, "-s", DEFAULTSEED);
    auto policy = std::execution::par;

    Timer<std::nano> timer;

    std::vector<int> libraryArr(size * 2);
    std::vector<int> gptArr(size * 2);

    // Additional vectors for testing
    std::vector<int> testVector1(size);
    std::vector<int> testVector2(size);
    defaultInit(testVector2, size);
    
    // Additional vectors for verification
    std::vector<int> libraryVerification(DEFAULTREPS);
    std::vector<int> gptVerification(DEFAULTREPS);

    auto stlLambda = [&] () {
        timer.start();
        int count = std::count(policy, testVector1.begin(), testVector1.end(), seed);
        timer.stop();

        libraryVerification.push_back(count);

        return timer.getElapsedTime();
    };

    auto gptLambda = [&] () {
        timer.start();
        int count = countElementsParallel(gptArr, seed);
        timer.stop();

        gptVerification.push_back(count);

        return timer.getElapsedTime();
    };

    // Print timing stats
    auto averageLibraryTime = runTests(testVector1, DEFAULTREPS, stlLambda, size, seed);
    printStats("Library Average: ", averageLibraryTime, "ns");
    auto averageGPTTime = runTests(testVector1, DEFAULTREPS, gptLambda, size, seed);
    printStats("ChatGPT Average: ", averageGPTTime, "ns");

    // Print verification stats
    bool validateGPT = containersAreEqual(libraryVerification, gptVerification);
    printStats("Validation: ", validateGPT ? "correct" : "incorrect", "");

    return 0;
}