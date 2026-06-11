cmake_minimum_required(VERSION 3.15)

file(GLOB ASTRA_SIM_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
)

file(GLOB ASTRA_SIM_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.h"
)

if(SIMAI_MODE STREQUAL "analytical")
    list(FILTER ASTRA_SIM_HEADERS EXCLUDE REGEX ".*SimAiFlowModelRdma\\.hh$")
    list(FILTER ASTRA_SIM_SOURCES EXCLUDE REGEX ".*SimAiFlowModelRdma\\.cc$")
    list(FILTER ASTRA_SIM_HEADERS EXCLUDE REGEX ".*BootStrapnet\\.hh$")
    list(FILTER ASTRA_SIM_SOURCES EXCLUDE REGEX ".*BootStrapnet\\.cc$")
    list(FILTER ASTRA_SIM_HEADERS EXCLUDE REGEX ".*PhyMultiThread\\.hh$")
    list(FILTER ASTRA_SIM_SOURCES EXCLUDE REGEX ".*PhyMultiThread\\.cc$")
endif()

add_library(AstraSim ${ASTRA_SIM_SOURCES} ${ASTRA_SIM_HEADERS})
target_compile_features(AstraSim PUBLIC cxx_std_17)
target_include_directories(AstraSim PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(USE_RDMA)
    target_compile_definitions(AstraSim PRIVATE PHY_RDMA PHY_MTP)

    find_package(MPI REQUIRED)
    target_link_libraries(AstraSim PUBLIC MPI::MPI_CXX)
endif()

if(SIMAI_MODE STREQUAL "analytical")
    target_compile_definitions(AstraSim PRIVATE ANALYTI)
endif()
