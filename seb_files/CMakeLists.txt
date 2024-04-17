cmake_minimum_required(VERSION 3.11)
project(group_project)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG main # Fetches the latest commit on the main branch
)

add_executable(gpu cuda/cuda_iscan/cuda_iscan.cu ../utils.hpp)
target_compile_features(gpu PRIVATE cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "70")
target_link_libraries(gpu PRIVATE CCCL::CCCL)

# Copy the job scripts
configure_file(../slurm/job-gpu job-gpu COPYONLY)
