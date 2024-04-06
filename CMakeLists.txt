cmake_minimum_required(VERSION 3.11)
project(group_project)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(gpu cuda/cuda_iscan/cuda_iscan.cu utils.hpp)
target_compile_features(gpu PRIVATE cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "70")

# Copy the job scripts
configure_file(slurm/job-gpu job-gpu COPYONLY)
