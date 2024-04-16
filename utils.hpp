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

enum class Policies 
{
    SEQUENTIAL,
    PARALLEL
};

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
            currentTime = Clock::now();
        }

        void stop()
        {
            previousTime = currentTime;
            currentTime = Clock::now();
        }

        double getElapsedTime()
        {
            std::chrono::duration<double, Ratio> elapsed = (currentTime - previousTime);
            return elapsed.count();
        }

        std::chrono::duration<double, Ratio> getElapsedTimeChrono()
        {
            return (currentTime - previousTime);
        }

    private:
        using Clock = std::chrono::high_resolution_clock;

        std::chrono::high_resolution_clock::time_point currentTime;
        std::chrono::high_resolution_clock::time_point previousTime;
};

template <typename Ratio = std::ratio<1>>
class Timer2
{
    public:
        Timer2() = default;

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
        using Clock = std::chrono::high_resolution_clock;

        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
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
void initRandomContainer(Container& container, size_t size, int seed)
{
    static std::random_device rd;
    static std::mt19937 gen(seed ? seed : rd());

    for (int i = 0; i < size; ++i)
    {
        container[i] = i;
    }

    std::shuffle(std::begin(container), std::end(container), gen);
}

template <typename Container>
bool containersAreEqual(Container& container1, Container& container2)
{
    return std::equal(std::begin(container1), std::end(container1), std::begin(container2));
}

template <typename Container, typename Lambda>
double runTests(Container& container, int repitions, Lambda& lambda, size_t size = 10'000, int seed = 1)
{
    Timer<std::nano> timer;
    double averageTime = 0;

    for (int i = 0; i < repitions; ++i)
    {
        initRandomContainer(container, size, seed);
        timer.start();
        lambda();
        timer.stop();

        averageTime += timer.getElapsedTime();
    }
    
    return averageTime / repitions;
}

template <typename T>
void printStats(std::string message, T value, std::string unit)
{
    std::cout << message << std::setw(20) << value << " " << unit << '\n';
}

