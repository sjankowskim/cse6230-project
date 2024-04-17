#include <algorithm>
#include <execution>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <ratio>
#include <vector>

#include "../utils.hpp"

#ifdef DEBUG
#include <set>
#include <mutex>
#include <thread>
#include <tbb/global_control.h>
#endif

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

void quickSort(std::vector<int>& vec, int low, int high) {
    if (low < high) {
        // Partitioning index
        int pivot = vec[high];  
        int i = (low - 1);

        for (int j = low; j <= high - 1; j++) {
            if (vec[j] < pivot) {
                i++;
                std::swap(vec[i], vec[j]);
            }
        }
        std::swap(vec[i + 1], vec[high]);
        int pi = i + 1;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quickSort(vec, low, pi - 1);
            }
            #pragma omp section
            {
                quickSort(vec, pi + 1, high);
            }
        }
    }
}

void sortVector(std::vector<int>& vec) {
    quickSort(vec, 0, vec.size() - 1);
}
/*-------------------------------*
 |         END SECTION           |
 *-------------------------------*/


int main(int argc, char** argv)
{
    size_t size = find_int_arg(argc, argv, "-n", DEFAULTSIZE);
    int seed = find_int_arg(argc, argv, "-s", DEFAULTSEED);
    auto policy = std::execution::seq;

    Timer<std::nano> timer;

    std::vector<int> libraryArr(size);
    std::vector<int> gptArr(size);

#ifdef DEBUG
    std::set<std::thread::id> thread_ids;
    std::mutex mutex;

    auto stlLambda = [&]()
    {
        timer.start();
        std::sort(policy, libraryArr.begin(), libraryArr.end(), [&](int i, int j)
                  {
            const std::lock_guard<std::mutex> lock(mutex);
            thread_ids.insert(std::this_thread::get_id());

            return i < j; });
        timer.stop();

        return timer.getElapsedTime();
    };
#else
    auto stlLambda = [&] () {
        timer.start();
        std::sort(policy, libraryArr.begin(), libraryArr.end());
        timer.stop();

        return timer.getElapsedTime();
    };
#endif

    auto gptLambda = [&] () {
        timer.start();
        sortVector(gptArr);
        timer.stop();

        return timer.getElapsedTime();
    };

    auto averageLibraryTime = runTests(libraryArr, DEFAULTREPS, stlLambda, size, seed);
    printStats("Library Average: ", averageLibraryTime, "ns");
    auto averageGPTTime = runTests(gptArr, DEFAULTREPS, gptLambda, size, seed);
    printStats("ChatGPT Average: ", averageGPTTime, "ns");


    bool validateGPT = containersAreEqual(libraryArr, gptArr);
    printStats("Validation: ", validateGPT ? "correct" : "incorrect", "");
#ifdef DEBUG
    std::cout << "Number of threads: " << thread_ids.size() << std::endl;
#endif

    return 0;
}