# CMake requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_definitions("-Wall -g")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Default build type" FORCE)
endif()

# Compiler requirement
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
        message(FATAL_ERROR "g++ (GNU) version should be greater than 5.3, but found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

project(SimAI)

set(NS3_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ns-3-alibabacloud/simulation")
set(ASTRA_SIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud/astra-sim")
set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(RESULT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/result")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create required directories
file(MAKE_DIRECTORY ${BUILD_DIR})
file(MAKE_DIRECTORY ${RESULT_DIR})

# Option to clean build and result directories
option(CLEAN_BUILD "Clean build directory" OFF)
option(CLEAN_RESULT "Clean result directory" OFF)

# Build types
option(BUILD_SIM "Build NS3 simulation" OFF)
option(BUILD_ANALYTICAL "Build analytical" OFF)
option(BUILD_PHY "Build physical" OFF)

# Other options
option(SYS_ASSERTS "Enable assert() in all profile with -UNDEBUG" OFF)

get_filename_component(BUILD_DIR_NAME ${BUILD_DIR} NAME)

function(ns3_export_file src dst)
    get_filename_component(_dst_abs "${dst}" ABSOLUTE)
    get_filename_component(dst_dir "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${dst_dir}")

    if(NS3_USE_RELATIVE_PATHS_SYMLINKS)
        # Compute a relative path from destination directory to source file.
        file(RELATIVE_PATH src_path "${dst_dir}" "${src}")
        # Ensure forward slashes for the preprocessor include
        file(TO_CMAKE_PATH "${src_path}" src_path)
    else()
        set(src_path "${src}")
    endif()

    if(NS3_EXPORT_HEADERS_AS_STUBS)
        # Remove destination file if it is a symlink
        if(IS_SYMLINK "${_dst_abs}")
            file(REMOVE "${dst}")
        endif()
        if("${src_path}" MATCHES "\\.(h|hh)$")
            file(GENERATE OUTPUT "${dst}" CONTENT "#include \"${src_path}\"\n")
        else()
            # Source files are just copied
            configure_file(${src} ${dst} COPYONLY)
        endif()
    else()
        if(NOT IS_SYMLINK "${_dst_abs}")
            # Create/overwrite a symbolic link.
            file(CREATE_LINK "${src_path}" "${dst}" SYMBOLIC)
        endif()
    endif()
endfunction()

if(CLEAN_BUILD)
    message(STATUS "Cleaning build directory: ${BUILD_DIR}")
    file(REMOVE_RECURSE "${BUILD_DIR}")
    file(MAKE_DIRECTORY ${BUILD_DIR})
    message(STATUS "Cleaning output directory: ${NS3_SRC_DIR}/build")
    file(REMOVE "${NS3_SRC_DIR}/.lock-ns3_linux_build")
    file(REMOVE_RECURSE "${NS3_SRC_DIR}/build")
    message(STATUS "Cleaning build directory: ${NS3_SRC_DIR}/${BUILD_DIR_NAME}")
    file(REMOVE_RECURSE "${NS3_SRC_DIR}/${BUILD_DIR_NAME}")
    message(STATUS "Cleaning bin directory: ${CMAKE_CURRENT_SOURCE_DIR}/bin")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endif()

if(CLEAN_RESULT)
    message(STATUS "Cleaning result directory: ${RESULT_DIR}")
    file(REMOVE_RECURSE ${RESULT_DIR})
    file(MAKE_DIRECTORY ${RESULT_DIR})
endif()

if(SYS_ASSERTS)
    add_compile_options(-UNDEBUG)
    message(STATUS "Asserts enabled for all profiles.")
endif()

if(BUILD_SIM)
    # AstraSim
    add_subdirectory(astra-sim-alibabacloud/build/astra_ns3)

    file(GLOB_RECURSE astra_sim_files LIST_DIRECTORIES false CONFIGURE_DEPENDS "${ASTRA_SIM_DIR}/*")
    list(FILTER astra_sim_files EXCLUDE REGEX "/network_frontend/")
    foreach(file ${astra_sim_files})
        string(REPLACE "${ASTRA_SIM_DIR}/" "astra-sim/" relative_path ${file})
        ns3_export_file("${file}" "${NS3_SRC_DIR}/src/applications/${relative_path}")
    endforeach()

    file(GLOB scratch_files LIST_DIRECTORIES false CONFIGURE_DEPENDS "${ASTRA_SIM_DIR}/network_frontend/ns3/*")
    foreach(file ${scratch_files})
        get_filename_component(filename ${file} NAME)
        ns3_export_file("${file}" "${NS3_SRC_DIR}/scratch/${filename}")
    endforeach()

    # Register original files as configure dependencies
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${astra_sim_files}
            ${scratch_files}
    )

    # NS3 options
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL "release")
        if (NOT DEFINED NS3_ASSERT)
            set(NS3_ASSERT OFF CACHE BOOL "Override NS3_ASSERT" FORCE)
        endif ()
        set(NS3_LOG OFF CACHE BOOL "Override NS3_LOG" FORCE)
    endif ()
    if (NOT DEFINED NS3_WARNINGS_AS_ERRORS)
        set(NS3_WARNINGS_AS_ERRORS OFF CACHE BOOL "Override NS3_WARNINGS_AS_ERRORS" FORCE)
    endif ()
    if (NOT DEFINED NS3_MTP)
        set(NS3_MTP ON CACHE BOOL "Override NS3_MTP" FORCE)
    endif ()

    # Remap original source files in the debug information of the target
    get_filename_component(COPIED_PREFIX_1 ${NS3_SRC_DIR}/src/applications/astra-sim REALPATH)
    get_filename_component(ORIGINAL_PREFIX_1 ${ASTRA_SIM_DIR} REALPATH)
    get_filename_component(COPIED_PREFIX_2 ${NS3_SRC_DIR}/scratch REALPATH)
    get_filename_component(ORIGINAL_PREFIX_2 ${ASTRA_SIM_DIR}/network_frontend/ns3 REALPATH)
    add_compile_options(
            -ffile-prefix-map=${COPIED_PREFIX_1}=${ORIGINAL_PREFIX_1}
            -ffile-prefix-map=${COPIED_PREFIX_2}=${ORIGINAL_PREFIX_2}
    )

    # NS3
    add_subdirectory(${NS3_SRC_DIR} ${NS3_SRC_DIR}/${BUILD_DIR_NAME})

    # Link binaries
    add_custom_target(create_ns3_symlink ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE:scratch_AstraSimNetwork>"
            "${CMAKE_CURRENT_SOURCE_DIR}/bin/SimAI_simulator"
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
