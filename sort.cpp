#include <algorithm>
#include <execution>
#include <iostream>
#include <omp.h>
#include <ratio>
#include <vector>

#include "utils.hpp"

constexpr int DEFAULTSIZE = 10'000'000;
constexpr int DEFAULTSEED = 100;
constexpr int DEFAULTREPS = 100;

// Function to swap two elements
void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

/// Partition using Lomuto's partition scheme
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];  // Pivot
    int i = (low - 1);      // Index of smaller element

    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than or equal to pivot
        if (arr[j] <= pivot) {
            i++; // Increment index of smaller element
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// Quicksort function
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        // pi is partitioning index, arr[p] is now at right place
        int pi = partition(arr, low, high);

        // Separately sort elements before partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv)
{
    int size = find_int_arg(argc, argv, "-n", DEFAULTSIZE);
    int seed = find_int_arg(argc, argv, "-s", DEFAULTSEED);
    int policy = find_int_arg(argc, argv, "-p", DEFAULTSEED);

    std::vector<int> systemArr;
    std::vector<int> gptArr;

    auto policy = std::execution::seq;

    auto gptLambda = [&] () {
        quickSort(gptArr, 0, gptArr.size() - 1);
    };

    auto libraryLambda = [&] () {
        std::sort(policy, systemArr.begin(), systemArr.end());
    };

    double averageGPTTime = runTests(gptArr, DEFAULTREPS, gptLambda, size);
    double averageLibraryTime = runTests(systemArr, DEFAULTREPS, libraryLambda, size);

    std::cout << "ChatGPT-4 implementation: " << averageGPTTime << " ns" << '\n';
    std::cout << "STL implementation: " << averageLibraryTime << " ns" << '\n';

    return 0;
}