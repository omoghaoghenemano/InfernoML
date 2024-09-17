cmake_minimum_required(VERSION 3.10)
project(InfernoML)

set(CMAKE_CXX_STANDARD 11)

# Include directories
include_directories(include)

# Source files
add_library(InfernoML STATIC
    src/algorithms/linear_regression.cpp
)

# Example executable
add_executable(linear_regression_example examples/linear_regression_example.cpp)
target_link_libraries(linear_regression_example PRIVATE InfernoML)
