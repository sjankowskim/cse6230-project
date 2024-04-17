#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <ratio>
#include <string>
#include <vector>

// ==========================
// Helper Classes and Enums
// ==========================

enum Algorithm {
    LIBRARY,
    GPT3,
    GPT4,
    COPILOT,
    GEMINI
};

// Default is in seconds
template <typename Ratio = std::ratio<1>>
class Timer
{
    public:
        Timer() = default;

        void start() 
        {
            start_time = Clock::now();
        }

        void stop()
        {
            end_time = Clock::now();
        }

        double getElapsedTime()
        {
            std::chrono::duration<double, Ratio> elapsed = (end_time - start_time);
            return elapsed.count();
        }

        std::chrono::duration<double, Ratio> getElapsedTimeChrono()
        {
            return (end_time - start_time);
        }

    private:
        using Clock = std::chrono::steady_clock;

        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
};


// =================
// Helper Functions
// =================

// if (find_arg_idx(argc, argv, "-h") >= 0) {
//     std::cout << "Options:" << std::endl;
//     std::cout << "-h: see this help" << std::endl;
//     std::cout << "-n <int>: set number of particles" << std::endl;
//     std::cout << "-o <filename>: set the output file name" << std::endl;
//     std::cout << "-s <int>: set particle initialization seed" << std::endl;
//     return 0;
// }

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

template <typename Container>
void defaultInit(Container& container, size_t size)
{
    for (int i = 0; i < size; ++i)
    {
        container[i] = i;
    }
}

template <typename Container>
void initRandomContainer(Container& container, size_t size, int seed, std::mt19937 gen)
{
    for (int i = 0; i < size; ++i)
    {
        container[i] = gen();
    }

    std::shuffle(std::begin(container), std::end(container), gen);
}

template <typename Container>
bool containersAreEqual(Container& container1, Container& container2)
{
    return std::equal(std::begin(container1), std::end(container1), std::begin(container2));
}

template <typename Container, typename Lambda>
double runTests(Container& container, int repetitions, Lambda& lambda, size_t size = 10'000, int seed = 1)
{
    // Intialize RNG 
    std::random_device rd;
    std::mt19937 gen(seed ? seed : rd());

    double averageTime = 0;

    for (int i = 0; i < repetitions; ++i)
    {
        initRandomContainer(container, size, seed, gen);
        
        averageTime += lambda();
    }
    
    return averageTime / repetitions;
}

template <typename T>
void printStats(std::string message, T value, std::string unit, int precision = 3)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << message << std::setw(20) << value << " " << unit << '\n';
}
