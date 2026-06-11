cmake_minimum_required(VERSION 3.15)

project(SimAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -g)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
        message(FATAL_ERROR
                "g++ (GNU) version should be greater than 5.3, but found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Default build type" FORCE)
endif()

set(NS3_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ns-3-alibabacloud/simulation")
set(ASTRA_SIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud/astra-sim")
set(RESULT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results")

file(MAKE_DIRECTORY "${RESULT_DIR}")

option(SYS_ASSERTS "Enable assert() in all profile with -UNDEBUG" OFF)

set(SIMAI_MODE "analytical" CACHE STRING "SimAI build mode")
set_property(CACHE SIMAI_MODE PROPERTY STRINGS analytical ns3 phy)

set(_valid_simai_modes analytical ns3 phy)
list(FIND _valid_simai_modes "${SIMAI_MODE}" _simai_mode_index)
if(_simai_mode_index EQUAL -1)
    message(FATAL_ERROR
            "Invalid SIMAI_MODE='${SIMAI_MODE}'. Expected one of: analytical, ns3, phy")
endif()

if(SYS_ASSERTS)
    add_compile_options(-UNDEBUG)
    message(STATUS "Asserts enabled for all profiles.")
endif()

if(SIMAI_MODE STREQUAL "ns3")
    add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud/build/astra_ns3"
    )
    set(target_name "scratch_AstraSimNetwork")
    set(symlink_name "SimAI_simulator")
elseif(SIMAI_MODE STREQUAL "analytical")
    add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud/build/simai_analytical"
    )
    set(target_name "SimAI_analytical")
    set(symlink_name "SimAI_analytical")
elseif(SIMAI_MODE STREQUAL "phy")
    set(USE_RDMA TRUE)
    add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud/build/simai_phy"
    )
    set(target_name "SimAI_phynet")
    set(symlink_name "SimAI_phynet")
else()
    message(FATAL_ERROR "Unknown SIMAI_MODE=${SIMAI_MODE}")
endif()

# Link binaries
add_custom_target(create_symlink ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/bin"
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:${target_name}>"
        "${CMAKE_SOURCE_DIR}/bin/${symlink_name}"
        DEPENDS ${target}
        COMMENT "Creating ${symlink_name} symlink..."
)
