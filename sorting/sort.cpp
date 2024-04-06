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

// Median-of-three pivot selection
int medianOfThree(std::vector<int>& arr, int low, int high) {
    int mid = low + (high - low) / 2;
    if (arr[mid] < arr[low])
        std::swap(arr[low], arr[mid]);
    if (arr[high] < arr[low])
        std::swap(arr[low], arr[high]);
    if (arr[mid] < arr[high])
        std::swap(arr[mid], arr[high]);
    return arr[high];
}

int partition(std::vector<int>& arr, int low, int high) {
    int pivot = medianOfThree(arr, low, high);
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Parallel version of the quickSort
void quickSortParallel(std::vector<int>& arr, int low, int high, int depth = 0) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Only create new threads if we haven't reached the maximum depth, calculated based on the number of processors
        if (depth < omp_get_max_threads()) {
            #pragma omp parallel sections
            {
                #pragma omp section
                { quickSortParallel(arr, low, pi - 1, depth + 1); }

                #pragma omp section
                { quickSortParallel(arr, pi + 1, high, depth + 1); }
            }
        } else { // If maximum depth reached, revert to sequential quicksort
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
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

    std::vector<int> systemArr;
    std::vector<int> gptArr;

#ifdef DEBUG
    std::set<std::thread::id> thread_ids;
    std::mutex mutex;

    auto stlLambda = [&] () {
        std::sort(policy, systemArr.begin(), systemArr.end(), [&](int i, int j){
            const std::lock_guard<std::mutex> lock(mutex);
            thread_ids.insert(std::this_thread::get_id());

            return i < j;
        });
    };
#else
    auto stlLambda = [&] () {
        std::sort(policy, systemArr.begin(), systemArr.end());
    };
#endif

    auto gptLambda = [&] () {
        quickSortParallel(gptArr, 0, gptArr.size() - 1);
    };

    std::cout << std::fixed << std::setprecision(3);
    double averageLibraryTime = runTests(systemArr, DEFAULTREPS, stlLambda, size, seed);
    printStats("Library Average: ", averageLibraryTime, "ns");
    double averageGPTTime = runTests(gptArr, DEFAULTREPS, gptLambda, size, seed);
    printStats("ChatGPT Average: ", averageGPTTime, "ns");

#ifdef DEBUG
    std::cout << "Number of threads: " << thread_ids.size() << std::endl;
#endif

    return 0;
}