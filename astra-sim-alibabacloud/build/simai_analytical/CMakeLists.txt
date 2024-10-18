# CMake requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compiler requirement
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
        message(FATAL_ERROR "g++ (GNU) version should be greater than 5.3, but found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()


# Setup project
project (AstraSim_Analytical)

# Compile AstraSim library
add_subdirectory("${PROJECT_SOURCE_DIR}/../../" AstraSim)

# Compile AnalyticalBackend binary
add_subdirectory ("${PROJECT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical" simai_analytical)
