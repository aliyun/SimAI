# CMake requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_definitions("-Wall -g")
SET(CMAKE_BUILD_TYPE "Debug")

# Compiler requirement
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
        message(FATAL_ERROR "g++ (GNU) version should be greater than 5.3, but found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

set(NS3_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ns-3-alibabacloud/simulation")
set(ASTRA_SIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud/astra-sim")
set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(RESULT_DIR "${CMAKE_CURRENT_SOURCE_DIR}}/result")

# Create required directories
file(MAKE_DIRECTORY ${BUILD_DIR})
file(MAKE_DIRECTORY ${RESULT_DIR})

# Option to clean build and result directories
option(CLEAN_BUILD "Clean build directory" OFF)
option(CLEAN_RESULT "Clean result directory" OFF)
option(BUILD_SIM "Build NS3 simulation" ON)
option(BUILD_ANALYTICAL "Build analytical" OFF)
option(BUILD_PHY "Build physical" OFF)

if(CLEAN_BUILD)
    message(STATUS "Cleaning build directory: ${BUILD_DIR}")
    file(REMOVE_RECURSE ${BUILD_DIR})
    file(REMOVE_RECURSE "${NS3_DIR}/simulation/build")
    file(REMOVE_RECURSE "${NS3_DIR}/simulation/cmake-cache")
    file(MAKE_DIRECTORY ${BUILD_DIR})
endif()

if(CLEAN_RESULT)
    message(STATUS "Cleaning result directory: ${RESULT_DIR}")
    file(REMOVE_RECURSE ${RESULT_DIR})
    file(MAKE_DIRECTORY ${RESULT_DIR})
endif()

if(BUILD_SIM)
    # AstraSim
    add_subdirectory(astra-sim-alibabacloud/build/astra_ns3)

    file(GLOB_RECURSE astra_sim_files CONFIGURE_DEPENDS "${ASTRA_SIM_DIR}/*")
    foreach(file ${astra_sim_files})
        string(REPLACE "${ASTRA_SIM_DIR}/" "astra-sim/" relative_path ${file})
        get_filename_component(output_dir ${NS3_SRC_DIR}/src/applications/${relative_path} DIRECTORY)
        file(MAKE_DIRECTORY ${output_dir})
        configure_file(${file} ${NS3_SRC_DIR}/src/applications/${relative_path} COPYONLY)
    endforeach()

    file(GLOB scratch_files
            CONFIGURE_DEPENDS
            ${ASTRA_SIM_DIR}/network_frontend/ns3/*)
    foreach(file ${scratch_files})
        get_filename_component(filename ${file} NAME)
        configure_file(${file} ${NS3_SRC_DIR}/scratch/${filename} COPYONLY)
    endforeach()

    # NS3
    add_subdirectory(${NS3_SRC_DIR} ${NS3_SRC_DIR}/cmake-cache)

    # Link binaries
    string(TOLOWER ${CMAKE_BUILD_TYPE} profile)
    add_custom_target(create_ns3_symlink
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${NS3_SRC_DIR}/build/scratch/ns3.36.1-AstraSimNetwork-${profile}"
            "${CMAKE_CURRENT_SOURCE_DIR}/bin/SimAI_simulator2"
            DEPENDS scratch_AstraSimNetwork
            COMMENT "Creating simulator symlink..."
    )
endif()

if(BUILD_ANALYTICAL)
    add_subdirectory(astra-sim-alibabacloud/build/simai_analytical)
endif()

if(BUILD_PHY)
    add_subdirectory(astra-sim-alibabacloud/build/simai_phy)
endif()
