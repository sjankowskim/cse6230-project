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

template <typename T>
void parallelCopyVector(const std::vector<T>& src, std::vector<T>& dest) {
    size_t n = src.size();
    dest.resize(n);  // Ensure destination is large enough.

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
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
    
    auto stlLambda = [&] () {
        timer.start();
        std::merge(policy, testVector1.begin(), testVector1.end(), testVector2.begin(), testVector2.end(), libraryArr.begin());
        timer.stop();

        return timer.getElapsedTime();
    };

    auto gptLambda = [&] () {
        timer.start();
        std::merge(policy, testVector1.begin(), testVector1.end(), testVector2.begin(), testVector2.end(), gptArr.begin());
        timer.stop();

        return timer.getElapsedTime();
    };

    auto averageLibraryTime = runTests(testVector1, DEFAULTREPS, stlLambda, size, seed);
    printStats("Library Average: ", averageLibraryTime, "ns");
    auto averageGPTTime = runTests(testVector1, DEFAULTREPS, gptLambda, size, seed);
    printStats("ChatGPT Average: ", averageGPTTime, "ns");

    bool validateGPT = containersAreEqual(libraryArr, gptArr);
    printStats("Validation: ", validateGPT ? "correct" : "incorrect", "");

    return 0;
}