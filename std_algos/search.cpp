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

template<typename T>
typename std::vector<T>::iterator parallelFindInVector(std::vector<T>& vec, const T& value) {
    typename std::vector<T>::iterator result = vec.end();
    
    #pragma omp parallel
    {
        typename std::vector<T>::iterator local_result = vec.end();
        
        #pragma omp for nowait
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] == value) {
                local_result = vec.begin() + i;
                #pragma omp flush
                break;
            }
        }

        #pragma omp critical
        {
            if (local_result < result) {
                result = local_result;
            }
        }
    }

    return result;
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

    int libraryFinds = 0;
    int gptFinds = 0;

    auto stlLambda = [&] () {
        timer.start();
        auto itr = std::find(policy, libraryArr.begin(), libraryArr.end(), seed);
        timer.stop();

        libraryFinds += (itr != libraryArr.end());

        return timer.getElapsedTime();
    };

    auto gptLambda = [&] () {
        timer.start();
        auto itr = parallelFindInVector(gptArr, seed);
        timer.stop();
        
        gptFinds += (itr != gptArr.end());

        return timer.getElapsedTime();
    };

    auto averageLibraryTime = runTests(libraryArr, DEFAULTREPS, stlLambda, size, seed);
    printStats("Library Average: ", averageLibraryTime, "ns");
    auto averageGPTTime = runTests(gptArr, DEFAULTREPS, gptLambda, size, seed);
    printStats("ChatGPT Average: ", averageGPTTime, "ns");

    return 0;
}