# CMake requirement
cmake_minimum_required(VERSION 3.15)


project(SimAI_phynet)
file(GLOB SOURCES "*.cc") 
file(GLOB HEADERS "*.h")    
include_directories("${PROJECT_SOURCE_DIR}/../../../")
include_directories("$ENV{MPI_INCLUDE_PATH}")

set(use_rdma ${USE_RDMA})
if(use_rdma)
    add_definitions(-DPHY_RDMA)
    message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
endif()
set(CMAKE_BUILD_TYPE Debug)
add_executable(SimAI_phynet ${SOURCES} ${HEADERS})

target_link_options(SimAI_phynet PRIVATE "-static-libstdc++")

if(use_rdma)
    target_link_libraries(SimAI_phynet AstraSim mpi ibverbs pthread)
endif()
