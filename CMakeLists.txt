cmake_minimum_required(VERSION 3.10)
project(InfernoML)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Define libraries
add_library(InfernoML STATIC
    src/algorithms/linear_regression.cpp
)
add_library(Kmeans STATIC
    src/algorithms/kmeans.cpp
)

add_library(Activations STATIC
    src/activation/activation_functions.cpp
)

# Define executables
add_executable(linear_regression_example examples/linear_regression_example.cpp)
add_executable(kmeans_functions_example examples/kmeans_functions_example.cpp)
add_executable(activation_functions_example examples/activation_functions_example.cpp)

# Link libraries to executables
target_link_libraries(linear_regression_example PRIVATE InfernoML)
target_link_libraries(activation_functions_example PRIVATE Activations)
target_link_libraries(kmeans_functions_example PRIVATE Kmeans)

# Ensure that include directories are added for the specific targets if necessary
target_include_directories(linear_regression_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(kmeans_functions_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(activation_functions_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
