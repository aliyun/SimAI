# CMake requirement
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_COMPILER "$ENV{MPI_BIN_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS} -std=c++11")

project (SimAi_Phy)

add_definitions(-DPHY_MTP)

# SimCCL mock version selection (matches ns3/analytical build default)
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
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/../../../SimCCL/src/mock/${MOCK_VERSION}" "${CMAKE_BINARY_DIR}/SimCCL/mock" SYMBOLIC)
include_directories("${CMAKE_BINARY_DIR}")

add_subdirectory("${PROJECT_SOURCE_DIR}/../../" AstraSim)

add_subdirectory ("${PROJECT_SOURCE_DIR}/../../astra-sim/network_frontend/phynet" simai_phynet)
