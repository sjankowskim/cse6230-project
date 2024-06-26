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

#pragma omp parallel  
{
    #pragma omp master
    {
    int num = omp_get_num_threads();
    std::cout << "Number of threads: " << num << std::endl;
    }
    
}

    Timer<std::nano> timer;

    std::vector<int> libraryArr(size);
    std::vector<int> gptArr(size);

    // Additional vectors for testing
    std::vector<int> testVector(size);
    
    // Additional vectors for verification
    std::vector<bool> libraryVerification(DEFAULTREPS);
    std::vector<bool> gptVerification(DEFAULTREPS);

    auto stlLambda = [&] () {
        timer.start();
        std::copy(policy, testVector.begin(), testVector.end(), libraryArr.begin());
        timer.stop();
        
        bool equals = std::equal(testVector.begin(), testVector.end(), libraryArr.begin());
        libraryVerification.push_back(equals);

        return timer.getElapsedTime();
    };

    auto gptLambda = [&] () {
        timer.start();
        parallelCopyVector(testVector, gptArr);
        timer.stop();

        bool equals = std::equal(testVector.begin(), testVector.end(), gptArr.begin());
        gptVerification.push_back(equals);

        return timer.getElapsedTime();
    };

    auto averageLibraryTime = runTests(testVector, DEFAULTREPS, stlLambda, size, seed);
    printStats("Library Average: ", averageLibraryTime, "ns");
    auto averageGPTTime = runTests(testVector, DEFAULTREPS, gptLambda, size, seed);
    printStats("ChatGPT Average: ", averageGPTTime, "ns");

    bool validateGPT = containersAreEqual(libraryVerification, gptVerification);
    printStats("Validation: ", validateGPT ? "correct" : "incorrect", "");

    

    return 0;
}