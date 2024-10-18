# CMake requirement
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_COMPILER "$ENV{MPI_BIN_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS} -std=c++11")

project (SimAi_Phy)

add_definitions(-DPHY_MTP)
add_subdirectory("${PROJECT_SOURCE_DIR}/../../" AstraSim)

add_subdirectory ("${PROJECT_SOURCE_DIR}/../../astra-sim/network_frontend/phynet" simai_phynet)
