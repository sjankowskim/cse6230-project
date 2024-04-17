set(CMAKE_CXX_STANDARD 17)

# Add compiler flags
add_compile_options(-O3 -ltbb )

# Get a list of all .cpp files in the current directory
file(GLOB SOURCES "*.cpp")

# Include the parent directory containing the header file
include_directories(${CMAKE_SOURCE_DIR}/..)

# Create an executable for each .cpp file
foreach(SOURCE_FILE ${SOURCES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE} ../utils.hpp)
endforeach()
