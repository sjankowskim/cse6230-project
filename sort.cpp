#include <algorithm>
#include <execution>
#include <iostream>
#include <omp.h>
#include <ratio>
#include <vector>

#include "utils.h"

constexpr int DEFAULTSIZE = 10'000'000;
constexpr int DEFAULTSEED = 1;
constexpr int DEFAULTREPS = 1;

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

void serialSort(std::vector<int>& arr)
{
    std::sort(arr.begin(), arr.end());
}

void parallelSort(std::vector<int>& arr)
{
    std::sort(std::execution::seq, arr.begin(), arr.end());
}

double runGPTTests(std::vector<int>& container, int repitions)
{
    Timer timer;
    double averageTime = 0;

    for (int i = 0; i < repitions; ++i)
    {
        initRandomContainer(container, DEFAULTSIZE, DEFAULTSEED);

        timer.start();
        quickSort(container, 0, container.size() - 1);
        timer.stop();

        averageTime += timer.getElapsedTime();
    }
    
    return averageTime / repitions;
}

double runLibraryTests(std::vector<int>& container, int repitions)
{
    Timer timer;
    double averageTime = 0;

    for (int i = 0; i < repitions; ++i)
    {
        initRandomContainer(container, DEFAULTSIZE, DEFAULTSEED);

        timer.start();
        serialSort(container);
        timer.stop();

        averageTime += timer.getElapsedTime();
    }
    
    return averageTime / repitions;
}


int main(int argc, char** argv)
{
    int size = find_int_arg(argc, argv, "-n", DEFAULTSIZE);
    int seed = find_int_arg(argc, argv, "-s", DEFAULTSEED);

    std::vector<int> systemArr;
    std::vector<int> gptArr;

    initRandomContainer(systemArr, size, seed);
    gptArr = systemArr;

    double averageGPTTime = runGPTTests(gptArr, DEFAULTREPS);
    double averageLibraryTime = runLibraryTests(gptArr, DEFAULTREPS);

    std::cout << "ChatGPT-4 implementation: " << averageGPTTime << " ms" << '\n';
    std::cout << "STL implementation: " << averageLibraryTime << " ms" << '\n';

    return 0;
}