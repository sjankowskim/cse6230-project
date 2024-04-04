#pragma once

#include <chrono>
#include <cstring>
#include <random>
#include <ratio>
#include <string>
#include <vector>

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
void initRandomContainer(Container& container, int size, int seed)
{
    static std::random_device rd;
    static std::mt19937 gen(seed ? seed : rd());

    container.reserve(size);

    for (int i = 0; i < size; ++i)
    {
        container.push_back(i);
    }

    std::shuffle(container.begin(), container.end(), gen);
}

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
