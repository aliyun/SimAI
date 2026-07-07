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

# SimCCL mock version selection (matches ns3 build default)
if(DEFINED ENV{SIMAI_NCCL_VERSION})
    set(MOCK_VERSION $ENV{SIMAI_NCCL_VERSION})
else()
    set(MOCK_VERSION "v2.30")
endif()

# Global include paths for SimCCL mock resolution
include_directories("${PROJECT_SOURCE_DIR}/../../")
include_directories("${PROJECT_SOURCE_DIR}/../../../")
# Create symlink: build_dir/SimCCL/mock/ -> actual mock version dir
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/SimCCL")
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/../../../SimCCL/mock/${MOCK_VERSION}" "${CMAKE_BINARY_DIR}/SimCCL/mock" SYMBOLIC)
include_directories("${CMAKE_BINARY_DIR}")

# Compile AstraSim library
add_subdirectory("${PROJECT_SOURCE_DIR}/../../" AstraSim)

# Compile AnalyticalBackend binary
add_subdirectory ("${PROJECT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical" simai_analytical)
