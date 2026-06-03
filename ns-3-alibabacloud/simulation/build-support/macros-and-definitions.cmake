# Copyright (c) 2017-2021 Universidade de Brasília
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation;
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA  02111-1307 USA
#
# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>

# Export compile time variable setting the directory to the NS3 root folder
add_definitions(-DPROJECT_SOURCE_PATH="${PROJECT_SOURCE_DIR}")

# Set INT128 as the default option for INT64X64 and register alternative
# implementations
set(NS3_INT64X64 "INT128" CACHE STRING "Int64x64 implementation")
set_property(CACHE NS3_INT64X64 PROPERTY STRINGS INT128 CAIRO DOUBLE)

# Purposefully hidden options:

# for ease of use, export all libraries and include directories to ns-3 module
# consumers by default
option(NS3_REEXPORT_THIRD_PARTY_LIBRARIES "Export all third-party libraries
and include directories to ns-3 module consumers" ON
)

# since we can't really do that safely from the CMake side
option(NS3_ENABLE_SUDO
       "Set executables ownership to root and enable the SUID flag" OFF
)

# Replace default CMake messages (logging) with custom colored messages as early
# as possible
include(${PROJECT_SOURCE_DIR}/build-support/3rd-party/colored-messages.cmake)

# WSLv1 doesn't support tap features
if(EXISTS "/proc/version")
  file(READ "/proc/version" CMAKE_LINUX_DISTRO)
  string(FIND ${CMAKE_LINUX_DISTRO} "Microsoft" res)
  if(res EQUAL -1)
    set(WSLv1 False)
  else()
    set(WSLv1 True)
  endif()
endif()

# Set Linux flag if on Linux
if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
  add_definitions(-D__LINUX__)
endif()

if(APPLE)
  add_definitions(-D__APPLE__)
endif()

set(cat_command cat)

if(CMAKE_XCODE_BUILD_SYSTEM)
  set(XCODE True)
else()
  set(XCODE False)
endif()

# Check the number of threads
include(ProcessorCount)
ProcessorCount(NumThreads)

# Output folders
if("${NS3_OUTPUT_DIRECTORY}" STREQUAL "")
  message(STATUS "Using default output directory ${PROJECT_SOURCE_DIR}/build")
  set(CMAKE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build) # default output
                                                          # folder
else()
  # Check if NS3_OUTPUT_DIRECTORY is a relative path
  set(absolute_ns3_output_directory "${NS3_OUTPUT_DIRECTORY}")
  if(NOT IS_ABSOLUTE ${NS3_OUTPUT_DIRECTORY})
    set(absolute_ns3_output_directory
        "${PROJECT_SOURCE_DIR}/${NS3_OUTPUT_DIRECTORY}"
    )
  endif()
  if(NOT (EXISTS ${absolute_ns3_output_directory}))
    message(
      STATUS
        "User-defined output directory \"${NS3_OUTPUT_DIRECTORY}\" doesn't exist. Trying to create it"
    )
    file(MAKE_DIRECTORY ${absolute_ns3_output_directory})
    if(NOT (EXISTS ${absolute_ns3_output_directory}))
      message(
        FATAL_ERROR
          "User-defined output directory \"${absolute_ns3_output_directory}\" could not be created. "
          "Try changing the value of NS3_OUTPUT_DIRECTORY"
      )
    endif()

    # If this directory is not inside the ns-3-dev folder, alert users tests may
    # break
    if(NOT ("${absolute_ns3_output_directory}" MATCHES "${PROJECT_SOURCE_DIR}"))
      message(
        WARNING
          "User-defined output directory \"${absolute_ns3_output_directory}\" is outside "
          " of the ns-3 directory ${PROJECT_SOURCE_DIR}, which will break some tests"
      )
    endif()
  endif()
  message(
    STATUS
      "User-defined output directory \"${absolute_ns3_output_directory}\" will be used"
  )
  set(CMAKE_OUTPUT_DIRECTORY ${absolute_ns3_output_directory})
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY})
set(CMAKE_HEADER_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY}/include/ns3)
set(THIRD_PARTY_DIRECTORY ${PROJECT_SOURCE_DIR}/3rd-party)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Get installation folder default values for each platform and include package
# configuration macro
include(GNUInstallDirs)
include(build-support/custom-modules/ns3-cmake-package.cmake)

if(${XCODE})
  # Is that so hard not to break people's CI, AAPL? Why would you output the
  # targets to a Debug/Release subfolder? Why?
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG}
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG}
        ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
    )
  endforeach()
endif()

# fPIC (position-independent code) and fPIE (position-independent executable)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# do not create a file-level dependency with shared libraries reducing
# unnecessary relinking
set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE)

# Identify compiler and check version
set(below_minimum_msg "compiler is below the minimum required version")
set(CLANG FALSE)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${AppleClang_MinVersion})
    message(
      FATAL_ERROR
        "Apple Clang ${CMAKE_CXX_COMPILER_VERSION} ${below_minimum_msg} ${AppleClang_MinVersion}"
    )
  endif()
  set(CLANG TRUE)
endif()

if((NOT CLANG) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${Clang_MinVersion})
    message(
      FATAL_ERROR
        "Clang ${CMAKE_CXX_COMPILER_VERSION} ${below_minimum_msg} ${Clang_MinVersion}"
    )
  endif()
  set(CLANG TRUE)
endif()

if(CLANG)
  if(${NS3_COLORED_OUTPUT} OR "$ENV{CLICOLOR}")
    add_definitions(-fcolor-diagnostics) # colorize clang++ output
  endif()
endif()

set(GCC FALSE)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${GNU_MinVersion})
    message(
      FATAL_ERROR
        "GNU ${CMAKE_CXX_COMPILER_VERSION} ${below_minimum_msg} ${GNU_MinVersion}"
    )
  endif()
  set(GCC TRUE)
  add_definitions(-fno-semantic-interposition)
  if(${NS3_COLORED_OUTPUT} OR "$ENV{CLICOLOR}")
    add_definitions(-fdiagnostics-color=always) # colorize g++ output
  endif()
endif()
unset(below_minimum_msg)

# Set compiler options and get command to force unused function linkage (useful
# for libraries)
set(CXX_UNSUPPORTED_STANDARDS 98 11 14)
set(CMAKE_CXX_STANDARD_MINIMUM 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LIB_AS_NEEDED_PRE)
set(LIB_AS_NEEDED_POST)
if(${GCC} AND NOT APPLE)
  # using GCC
  set(LIB_AS_NEEDED_PRE -Wl,--no-as-needed)
  set(LIB_AS_NEEDED_POST -Wl,--as-needed)
  set(LIB_AS_NEEDED_PRE_STATIC -Wl,--whole-archive,-Bstatic)
  set(LIB_AS_NEEDED_POST_STATIC -Wl,--no-whole-archive)
  set(LIB_AS_NEEDED_POST_STATIC_DYN -Wl,-Bdynamic,--no-whole-archive)
endif()

if(${CLANG} AND APPLE)
  # using Clang set(LIB_AS_NEEDED_PRE -all_load)
  set(LIB_AS_NEEDED_POST)
endif()

macro(SUBDIRLIST result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

macro(library_target_name libname targetname)
  set(${targetname} lib${libname})
endmacro()

macro(clear_global_cached_variables)
  # clear cache variables
  unset(build_profile CACHE)
  unset(build_profile_suffix CACHE)
  unset(lib-ns3-static-objs CACHE)
  unset(ns3-contrib-libs CACHE)
  unset(ns3-example-folders CACHE)
  unset(ns3-execs CACHE)
  unset(ns3-execs-py CACHE)
  unset(ns3-external-libs CACHE)
  unset(ns3-headers-to-module-map CACHE)
  unset(ns3-libs CACHE)
  unset(ns3-libs-tests CACHE)
  unset(ns3-python-bindings-modules CACHE)
  mark_as_advanced(
    build_profile
    build_profile_suffix
    lib-ns3-static-objs
    ns3-contrib-libs
    ns3-example-folders
    ns3-execs
    ns3-execs-py
    ns3-external-libs
    ns3-headers-to-module-map
    ns3-libs
    ns3-libs-tests
    ns3-python-bindings-modules
  )
endmacro()

# function used to search for package and program dependencies than store list
# of missing dependencies in the list whose name is stored in missing_deps
function(check_deps package_deps program_deps missing_deps)
  set(local_missing_deps)
  # Search for package dependencies
  foreach(package ${package_deps})
    find_package(${package})
    if(NOT ${${package}_FOUND})
      list(APPEND local_missing_deps ${package})
    endif()
  endforeach()

  # And for program dependencies
  foreach(program ${program_deps})
    # CMake likes to cache find_* to speed things up, so we can't reuse names
    # here or it won't check other dependencies
    string(TOUPPER ${program} upper_${program})
    mark_as_advanced(${upper_${program}})
    find_program(${upper_${program}} ${program})
    if("${${upper_${program}}}" STREQUAL "${upper_${program}}-NOTFOUND")
      list(APPEND local_missing_deps ${program})
    endif()
  endforeach()

  # Store list of missing dependencies in the parent scope
  set(${missing_deps} ${local_missing_deps} PARENT_SCOPE)
endfunction()

# process all options passed in main cmakeLists
macro(process_options)
  clear_global_cached_variables()

  # make sure to default to RelWithDebInfo if no build type is specified
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "default" CACHE STRING "Choose the type of build."
                                         FORCE
    )
    set(NS3_ASSERT ON CACHE BOOL "Enable assert on failure" FORCE)
    set(NS3_LOG ON CACHE BOOL "Enable logging to be built" FORCE)
    set(NS3_WARNINGS_AS_ERRORS OFF
        CACHE BOOL "Treat warnings as errors. Requires NS3_WARNINGS=ON" FORCE
    )
  endif()

  # process debug switch Used in build-profile-test-suite
  string(TOLOWER ${CMAKE_BUILD_TYPE} cmakeBuildType)
  set(build_profile "${cmakeBuildType}" CACHE INTERNAL "")
  if(${cmakeBuildType} STREQUAL "debug")
    add_definitions(-DNS3_BUILD_PROFILE_DEBUG)
  elseif(${cmakeBuildType} STREQUAL "relwithdebinfo" OR ${cmakeBuildType}
                                                        STREQUAL "default"
  )
    set(cmakeBuildType relwithdebinfo)
    set(CMAKE_CXX_FLAGS_DEFAULT ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    add_definitions(-DNS3_BUILD_PROFILE_DEBUG)
  elseif(${cmakeBuildType} STREQUAL "release")
    if(${NS3_NATIVE_OPTIMIZATIONS})
      add_definitions(-DNS3_BUILD_PROFILE_OPTIMIZED)
      set(build_profile "optimized" CACHE INTERNAL "")
    else()
      add_definitions(-DNS3_BUILD_PROFILE_RELEASE)
    endif()
  else()
    add_definitions(-DNS3_BUILD_PROFILE_RELEASE)
  endif()

  # Enable examples if activated via command line (NS3_EXAMPLES) or ns3rc config
  # file
  set(ENABLE_EXAMPLES OFF)
  if(${NS3_EXAMPLES} OR ${ns3rc_examples_enabled})
    set(ENABLE_EXAMPLES ON)
  endif()

  # Enable examples if activated via command line (NS3_TESTS) or ns3rc config
  # file
  set(ENABLE_TESTS OFF)
  if(${NS3_TESTS} OR ${ns3rc_tests_enabled})
    set(ENABLE_TESTS ON)
    enable_testing()
  endif()

  set(profiles_without_suffixes release)
  set(build_profile_suffix "" CACHE INTERNAL "")
  if(NOT (${build_profile} IN_LIST profiles_without_suffixes))
    set(build_profile_suffix -${build_profile} CACHE INTERNAL "")
  endif()

  # Set warning level and warning as errors
  if(${NS3_WARNINGS})
    if(MSVC)
      add_compile_options(/W3) # /W4 = -Wall + -Wextra
      if(${NS3_WARNINGS_AS_ERRORS})
        add_compile_options(/WX)
      endif()
    else()
      add_compile_options(-Wall) # -Wextra
      if(${NS3_WARNINGS_AS_ERRORS})
        add_compile_options(-Werror -Wno-error=deprecated-declarations)
      endif()
    endif()
  endif()

  include(build-support/custom-modules/ns3-versioning.cmake)
  set(ENABLE_BUILD_VERSION False)
  configure_embedded_version()

  if(${NS3_CLANG_FORMAT})
    find_program(CLANG_FORMAT clang-format)
    if("${CLANG_FORMAT}" STREQUAL "CLANG_FORMAT-NOTFOUND")
      message(${HIGHLIGHTED_STATUS} "Proceeding without clang-format")
    else()
      file(
        GLOB_RECURSE
        ALL_CXX_SOURCE_FILES
        src/*.cc
        src/*.h
        examples/*.cc
        examples/*.h
        utils/*.cc
        utils/*.h
        scratch/*.cc
        scratch/*.h
      )
      add_custom_target(
        clang-format COMMAND clang-format -style=file -i
                             ${ALL_CXX_SOURCE_FILES}
      )
      unset(ALL_CXX_SOURCE_FILES)
    endif()
  endif()

  if(${NS3_CLANG_TIDY})
    find_program(CLANG_TIDY clang-tidy)
    if("${CLANG_TIDY}" STREQUAL "CLANG_TIDY-NOTFOUND")
      message(${HIGHLIGHTED_STATUS}
              "Proceeding without clang-tidy static analysis"
      )
    else()
      set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
    endif()
  endif()

  if(${NS3_CLANG_TIMETRACE})
    if(${CLANG})
      # Fetch and build clang build analyzer
      include(ExternalProject)
      ExternalProject_Add(
        ClangBuildAnalyzer
        GIT_REPOSITORY "https://github.com/aras-p/ClangBuildAnalyzer.git"
        GIT_TAG "47406981a1c5a89e8f8c62802b924c3e163e7cb4"
        INSTALL_COMMAND cmake -E copy_if_different ClangBuildAnalyzer
                        ${PROJECT_BINARY_DIR}
      )

      # Add compiler flag and create target to do everything automatically
      add_definitions(-ftime-trace)
      add_custom_target(
        timeTraceReport
        COMMAND
          ${PROJECT_BINARY_DIR}/ClangBuildAnalyzer --all ${PROJECT_BINARY_DIR}
          ${PROJECT_BINARY_DIR}/clangBuildAnalyzerReport.bin
        COMMAND
          ${PROJECT_BINARY_DIR}/ClangBuildAnalyzer --analyze
          ${PROJECT_BINARY_DIR}/clangBuildAnalyzerReport.bin >
          ${PROJECT_SOURCE_DIR}/ClangBuildAnalyzerReport.txt
        DEPENDS ClangBuildAnalyzer
      )
    else()
      message(
        FATAL_ERROR
          "TimeTrace is a Clang feature, but you're using a different compiler."
      )
    endif()
  endif()

  mark_as_advanced(CMAKE_FORMAT_PROGRAM)
  find_program(CMAKE_FORMAT_PROGRAM cmake-format HINTS ~/.local/bin)
  if("${CMAKE_FORMAT_PROGRAM}" STREQUAL "CMAKE_FORMAT_PROGRAM-NOTFOUND")
    message(${HIGHLIGHTED_STATUS} "Proceeding without cmake-format")
  else()
    file(GLOB_RECURSE MODULES_CMAKE_FILES src/**/CMakeLists.txt
         contrib/**/CMakeLists.txt examples/**/CMakeLists.txt
    )
    file(
      GLOB
      INTERNAL_CMAKE_FILES
      CMakeLists.txt
      utils/**/CMakeLists.txt
      src/CMakeLists.txt
      build-support/**/*.cmake
      build-support/*.cmake
    )
    add_custom_target(
      cmake-format
      COMMAND
        ${CMAKE_FORMAT_PROGRAM} -c
        ${PROJECT_SOURCE_DIR}/build-support/cmake-format.txt -i
        ${INTERNAL_CMAKE_FILES}
      COMMAND
        ${CMAKE_FORMAT_PROGRAM} -c
        ${PROJECT_SOURCE_DIR}/build-support/cmake-format-modules.txt -i
        ${MODULES_CMAKE_FILES}
    )
    unset(MODULES_CMAKE_FILES)
    unset(INTERNAL_CMAKE_FILES)
  endif()

  # If the user has not set a CXX standard version, assume the minimum
  if(NOT "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD_MINIMUM})
  endif()

  # Search standard provided by the user in the unsupported standards list
  list(FIND CXX_UNSUPPORTED_STANDARDS ${CMAKE_CXX_STANDARD} unsupported)
  if(${unsupported} GREATER -1)
    message(
      FATAL_ERROR
        "You're trying to use the unsupported C++ ${CMAKE_CXX_STANDARD}.\n"
        "Try -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD_MINIMUM}."
    )
  endif()

  if(${NS3_DES_METRICS})
    add_definitions(-DENABLE_DES_METRICS)
  endif()

  if(${NS3_SANITIZE} AND ${NS3_SANITIZE_MEMORY})
    message(
      FATAL_ERROR
        "The memory sanitizer can't be used with other sanitizers. Disable one of them."
    )
  endif()

  if(${NS3_SANITIZE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,leak,undefined")
  endif()

  if(${NS3_SANITIZE_MEMORY})
    if(${CLANG})
      set(blacklistfile ${PROJECT_SOURCE_DIR}/memory-sanitizer-blacklist.txt)
      if(EXISTS ${blacklistfile})
        set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -fsanitize=memory -fsanitize-blacklist=${blacklistfile}"
        )
      else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
      endif()

      if(NOT ($ENV{MSAN_OPTIONS} MATCHES "strict_memcmp=0"))
        message(
          WARNING
            "Please export MSAN_OPTIONS=strict_memcmp=0 "
            "and call the generated buildsystem directly to proceed.\n"
            "Trying to build with cmake or IDEs will probably fail since"
            "CMake can't export environment variables to its parent process."
        )
      endif()
      unset(blacklistfile)
    else()
      message(FATAL_ERROR "The memory sanitizer is only supported by Clang")
    endif()
  endif()

  set(ENABLE_SQLITE False)
  if(${NS3_SQLITE})
    # find_package(SQLite3 QUIET) # unsupported in CMake 3.10 We emulate the
    # behavior of find_package below
    find_external_library(
      DEPENDENCY_NAME SQLite3 HEADER_NAME sqlite3.h LIBRARY_NAME sqlite3
    )

    if(${SQLite3_FOUND})
      set(ENABLE_SQLITE True)
      add_definitions(-DHAVE_SQLITE3)
      include_directories(${SQLite3_INCLUDE_DIRS})
    else()
      message(${HIGHLIGHTED_STATUS} "SQLite was not found")
    endif()
  endif()

  if(${NS3_NATIVE_OPTIMIZATIONS} AND ${GCC})
    add_compile_options(-march=native -mtune=native)
  endif()

  if(${NS3_LINK_TIME_OPTIMIZATION})
    # Link-time optimization (LTO) if available
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_AVAILABLE OUTPUT output)
    if(LTO_AVAILABLE)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
      message(STATUS "Link-time optimization (LTO) is supported.")
    else()
      message(
        STATUS "Link-time optimization (LTO) is not supported: ${output}."
      )
    endif()
  endif()

  if(${NS3_LINK_WHAT_YOU_USE})
    set(CMAKE_LINK_WHAT_YOU_USE TRUE)
  else()
    set(CMAKE_LINK_WHAT_YOU_USE FALSE)
  endif()

  if(${NS3_INCLUDE_WHAT_YOU_USE})
    # Before using, install iwyu, run "strings /usr/bin/iwyu | grep LLVM" to
    # find the appropriate clang version and install it. If you don't do that,
    # it will fail. Worse than that: it will fail silently. We use the wrapper
    # to get a iwyu.log file in the ns-3-dev folder.
    find_program(INCLUDE_WHAT_YOU_USE_PROG iwyu)
    if("${INCLUDE_WHAT_YOU_USE_PROG}" STREQUAL
       "INCLUDE_WHAT_YOU_USE_PROG-NOTFOUND"
    )
      message(FATAL_ERROR "iwyu (include-what-you-use) was not found.")
    endif()
    message(STATUS "iwyu is enabled")
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE
        ${PROJECT_SOURCE_DIR}/build-support/iwyu-wrapper.sh;${PROJECT_SOURCE_DIR}
    )
  else()
    unset(CMAKE_CXX_INCLUDE_WHAT_YOU_USE)
  endif()

  if(${XCODE})
    if(${NS3_STATIC} OR ${NS3_MONOLIB})
      message(
        FATAL_ERROR
          "Xcode doesn't play nicely with CMake object libraries,"
          "and those are used for NS3_STATIC and NS3_MONOLIB.\n"
          "Disable them or try a different generator"
      )
    endif()
    set(CMAKE_XCODE_GENERATE_TOP_LEVEL_PROJECT_ONLY ON)
    set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED NO)
    set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "")
  else()
    unset(CMAKE_XCODE_GENERATE_TOP_LEVEL_PROJECT_ONLY)
    unset(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED)
    unset(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY)
  endif()

  # Set common include folder (./build/include, where we find ns3/core-module.h)
  include_directories(${CMAKE_OUTPUT_DIRECTORY}/include)

  # find required dependencies
  list(APPEND CMAKE_MODULE_PATH
       "${PROJECT_SOURCE_DIR}/build-support/custom-modules"
  )
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/build-support/3rd-party")

  # GTK3 Don't search for it if you don't have it installed, as it take an
  # insane amount of time
  if(${NS3_GTK3})
    find_package(HarfBuzz QUIET)
    if(NOT ${HarfBuzz_FOUND})
      message(${HIGHLIGHTED_STATUS}
              "Harfbuzz is required by GTK3 and was not found."
      )
    else()
      set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
      find_package(GTK3 QUIET)
      unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS CACHE)
      if(NOT ${GTK3_FOUND})
        message(${HIGHLIGHTED_STATUS}
                "GTK3 was not found. Continuing without it."
        )
      else()
        if(${GTK3_VERSION} VERSION_LESS 3.22)
          set(GTK3_FOUND FALSE)
          message(${HIGHLIGHTED_STATUS}
                  "GTK3 found with incompatible version ${GTK3_VERSION}"
          )
        else()
          message(STATUS "GTK3 was found.")
          include_directories(${GTK3_INCLUDE_DIRS} ${HarfBuzz_INCLUDE_DIRS})
        endif()
      endif()

    endif()
  endif()

  if(${NS3_STATIC})
    # Warn users that they may be using shared libraries, which won't produce a
    # standalone static library
    set(ENABLE_REALTIME FALSE)
    set(HAVE_RT FALSE)
    message(
      WARNING "Statically linking 3rd party libraries have not been tested.\n"
              "Disable Brite, Click, Gtk, GSL, Mpi, Openflow and SQLite"
              " if you want a standalone static ns-3 library."
    )
  else()
    find_package(LibXml2 QUIET)
    if(NOT ${LIBXML2_FOUND})
      message(${HIGHLIGHTED_STATUS}
              "LibXML2 was not found. Continuing without it."
      )
    else()
      message(STATUS "LibXML2 was found.")
      add_definitions(-DHAVE_LIBXML2)
      include_directories(${LIBXML2_INCLUDE_DIR})
    endif()

    # LibRT
    mark_as_advanced(LIBRT)
    set(ENABLE_REALTIME FALSE)
    if(${NS3_REALTIME})
      if(APPLE)
        message(
          STATUS "Lib RT is not supported on Mac OS X. Continuing without it."
        )
      else()
        find_library(LIBRT rt QUIET)
        if(NOT ${LIBRT_FOUND})
          message(FATAL_ERROR "LibRT was not found.")
        else()
          message(STATUS "LibRT was found.")
          set(ENABLE_REALTIME TRUE)
          set(HAVE_RT TRUE) # for core-config.h
        endif()
      endif()
    endif()
  endif()

  set(THREADS_PREFER_PTHREAD_FLAG)
  find_package(Threads QUIET)
  if(NOT ${Threads_FOUND})
    message(FATAL_ERROR Threads are required by ns-3)
  endif()

  set(Python3_LIBRARIES)
  set(Python3_EXECUTABLE)
  set(Python3_FOUND FALSE)
  set(Python3_INCLUDE_DIRS)
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
    find_package(Python3 COMPONENTS Interpreter Development)
  else()
    # cmake-format: off
    set(Python_ADDITIONAL_VERSIONS 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9)
    # cmake-format: on
    find_package(PythonInterp)
    find_package(PythonLibs)

    # Move deprecated results into the FindPython3 resulting variables
    set(Python3_Interpreter_FOUND ${PYTHONINTERP_FOUND})
    set(Python3_Development_FOUND ${PYTHONLIBS_FOUND})
    if(${PYTHONINTERP_FOUND})
      set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
      set(Python3_FOUND TRUE)
    endif()
    if(${PYTHONLIBS_FOUND})
      set(Python3_LIBRARIES ${PYTHON_LIBRARIES})
      set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
    endif()
  endif()

  # Check if both Python interpreter and development libraries were found
  if(${Python3_Interpreter_FOUND})
    if(${Python3_Development_FOUND})
      set(Python3_FOUND TRUE)
      if(APPLE)
        # Apple is very weird and there could be a lot of conflicting python
        # versions which can generate conflicting rpaths preventing the python
        # bindings from working

        # To work around, we extract the /path/to/Frameworks from the library
        # path
        list(GET Python3_LIBRARIES 0 pylib)
        string(REGEX REPLACE "(.*Frameworks)/Python(3.|.)framework.*" "\\1"
                             DEVELOPER_DIR ${pylib}
        )
        if("${DEVELOPER_DIR}" MATCHES "Frameworks")
          set(CMAKE_BUILD_RPATH "${DEVELOPER_DIR}" CACHE STRING "")
          set(CMAKE_INSTALL_RPATH "${DEVELOPER_DIR}" CACHE STRING "")
        endif()
      endif()
      include_directories(${Python3_INCLUDE_DIRS})
    else()
      message(${HIGHLIGHTED_STATUS}
              "Python: development libraries were not found"
      )
    endif()
  else()
    message(
      ${HIGHLIGHTED_STATUS}
      "Python: an incompatible version of Python was found, python bindings will be disabled"
    )
  endif()

  set(ENABLE_PYTHON_BINDINGS OFF)
  if(${NS3_PYTHON_BINDINGS})
    if(NOT ${Python3_FOUND})
      message(
        FATAL_ERROR
          "Bindings: python bindings require Python, but it could not be found"
      )
    else()
      check_python_packages("pybindgen" missing_packages)
      if(missing_packages)
        message(
          FATAL_ERROR
            "Bindings: python bindings disabled due to the following missing dependencies: ${missing_packages}"
        )
      else()
        set(ENABLE_PYTHON_BINDINGS ON)

        set(destination_dir ${CMAKE_OUTPUT_DIRECTORY}/bindings/python/ns)
        configure_file(
          bindings/python/ns__init__.py ${destination_dir}/__init__.py COPYONLY
        )
      endif()
    endif()
  endif()

  # Disable the below warning from bindings built in debug mode with clang++:
  # "expression with side effects will be evaluated despite being used as an
  # operand to 'typeid'"
  if(${ENABLE_PYTHON_BINDINGS} AND ${CLANG})
    add_compile_options(-Wno-potentially-evaluated-expression)
  endif()

  set(ENABLE_SCAN_PYTHON_BINDINGS OFF)
  if(${NS3_SCAN_PYTHON_BINDINGS})
    if(NOT ${Python3_FOUND})
      message(
        FATAL_ERROR
          "Bindings: scanning python bindings require Python, but it could not be found"
      )
    else()
      # Check if pybindgen, pygccxml, cxxfilt and castxml are installed
      check_python_packages(
        "pybindgen;pygccxml;cxxfilt;castxml" missing_packages
      )

      # If castxml has not been found via python import, fallback to searching
      # the binary
      if(castxml IN_LIST missing_packages)
        mark_as_advanced(CASTXML)
        find_program(CASTXML castxml)
        if(NOT ("${CASTXML}" STREQUAL "CASTXML-NOTFOUND"))
          list(REMOVE_ITEM missing_packages castxml)
        endif()
      endif()

      # If packages were not found, print message
      if(missing_packages)
        message(
          FATAL_ERROR
            "Bindings: scanning of python bindings disabled due to the following missing dependencies: ${missing_packages}"
        )
      else()
        set(ENABLE_SCAN_PYTHON_BINDINGS ON)
        # empty scan target that will depend on other module scan targets to
        # scan all of them
        add_custom_target(apiscan-all)
      endif()
    endif()
  endif()

  set(ENABLE_VISUALIZER FALSE)
  if(${NS3_VISUALIZER})
    if((NOT ${ENABLE_PYTHON_BINDINGS}) OR (NOT ${Python3_FOUND}))
      message(${HIGHLIGHTED_STATUS} "Visualizer requires Python bindings")
    else()
      set(ENABLE_VISUALIZER TRUE)
    endif()
  endif()

  if(${NS3_COVERAGE} AND (NOT ${ENABLE_TESTS} OR NOT ${ENABLE_EXAMPLES}))
    message(
      FATAL_ERROR
        "Code coverage requires examples and tests.\nTry reconfiguring CMake with -DNS3_TESTS=ON -DNS3_EXAMPLES=ON"
    )
  endif()

  if(${ENABLE_TESTS})
    add_custom_target(all-test-targets)

    # Create a custom target to run test.py --no-build Target is also used to
    # produce code coverage output
    add_custom_target(
      run_test_py
      COMMAND ${Python3_EXECUTABLE} test.py --no-build
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS all-test-targets
    )
    if(${ENABLE_EXAMPLES})
      include(build-support/custom-modules/ns3-coverage.cmake)
    endif()
  endif()

  # Process config-store-config
  configure_file(
    build-support/config-store-config-template.h
    ${CMAKE_HEADER_OUTPUT_DIRECTORY}/config-store-config.h
  )

  set(ENABLE_MPI FALSE)
  if(${NS3_MPI})
    find_package(MPI QUIET)
    if(NOT ${MPI_FOUND})
      message(FATAL_ERROR "MPI was not found.")
    else()
      message(STATUS "MPI was found.")
      add_definitions(-DNS3_MPI)
      include_directories(${MPI_CXX_INCLUDE_DIRS})
      set(ENABLE_MPI TRUE)
    endif()
  endif()

  set(ENABLE_MTP FALSE)
  if(${NS3_MTP})
    add_definitions(-DNS3_MTP)
    set(ENABLE_MTP TRUE)
  endif()

  if(${NS3_VERBOSE})
    set_property(GLOBAL PROPERTY TARGET_MESSAGES TRUE)
    set(CMAKE_FIND_DEBUG_MODE TRUE)
    set(CMAKE_VERBOSE_MAKEFILE TRUE CACHE INTERNAL "")
  else()
    set_property(GLOBAL PROPERTY TARGET_MESSAGES OFF)
    unset(CMAKE_FIND_DEBUG_MODE)
    unset(CMAKE_VERBOSE_MAKEFILE CACHE)
  endif()

  mark_as_advanced(Boost_INCLUDE_DIR)
  find_package(Boost)
  if(${Boost_FOUND})
    include_directories(${Boost_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})
  endif()

  if(${NS3_GSL})
    find_package(GSL QUIET)
    if(NOT ${GSL_FOUND})
      message(${HIGHLIGHTED_STATUS} "GSL was not found. Continuing without it.")
    else()
      message(STATUS "GSL was found.")
      add_definitions(-DHAVE_GSL)
      include_directories(${GSL_INCLUDE_DIRS})
    endif()
  endif()

  if(${NS3_GNUPLOT})
    find_package(Gnuplot-ios) # Not sure what package would contain the correct
                              # header/library
    if(NOT ${GNUPLOT_FOUND})
      message(${HIGHLIGHTED_STATUS}
              "GNUPLOT was not found. Continuing without it."
      )
    else()
      message(STATUS "GNUPLOT was found.")
      include_directories(${GNUPLOT_INCLUDE_DIRS})
      link_directories(${GNUPLOT_LIBRARY})
    endif()
  endif()

  # checking for documentation dependencies and creating targets

  # First we check for doxygen dependencies
  mark_as_advanced(DOXYGEN)
  check_deps("" "doxygen;dot;dia" doxygen_docs_missing_deps)
  if(doxygen_docs_missing_deps)
    message(
      ${HIGHLIGHTED_STATUS}
      "docs: doxygen documentation not enabled due to missing dependencies: ${doxygen_docs_missing_deps}"
    )
    # cmake-format: off
    set(doxygen_missing_msg
        echo The following Doxygen dependencies are missing: ${doxygen_docs_missing_deps}.
            Reconfigure the project after installing them.
    )
    # cmake-format: on

    # Create stub targets to inform users about missing dependencies
    add_custom_target(
      run-print-introspected-doxygen COMMAND ${doxygen_missing_msg}
    )
    add_custom_target(
      run-introspected-command-line COMMAND ${doxygen_missing_msg}
    )
    add_custom_target(
      assemble-introspected-command-line COMMAND ${doxygen_missing_msg}
    )
    add_custom_target(update_doxygen_version COMMAND ${doxygen_missing_msg})
    add_custom_target(doxygen COMMAND ${doxygen_missing_msg})
    add_custom_target(doxygen-no-build COMMAND ${doxygen_missing_msg})
  else()
    # We checked this already exists, but we need the path to the executable
    find_package(Doxygen QUIET)

    # Get introspected doxygen
    add_custom_target(
      run-print-introspected-doxygen
      COMMAND
        ${CMAKE_OUTPUT_DIRECTORY}/utils/ns${NS3_VER}-print-introspected-doxygen${build_profile_suffix}
        > ${PROJECT_SOURCE_DIR}/doc/introspected-doxygen.h
      COMMAND
        ${CMAKE_OUTPUT_DIRECTORY}/utils/ns${NS3_VER}-print-introspected-doxygen${build_profile_suffix}
        --output-text > ${PROJECT_SOURCE_DIR}/doc/ns3-object.txt
      DEPENDS print-introspected-doxygen
    )
    add_custom_target(
      run-introspected-command-line
      COMMAND ${CMAKE_COMMAND} -E env NS_COMMANDLINE_INTROSPECTION=..
              ${Python3_EXECUTABLE} ./test.py --no-build --constrain=example
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS all-test-targets # all-test-targets only exists if ENABLE_TESTS is
                               # set to ON
    )

    file(
      WRITE ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h
      "/* This file is automatically generated by
  CommandLine::PrintDoxygenUsage() from the CommandLine configuration
  in various example programs.  Do not edit this file!  Edit the
  CommandLine configuration in those files instead.
  */
  \n"
    )
    add_custom_target(
      assemble-introspected-command-line
      # works on CMake 3.18 or newer > COMMAND ${CMAKE_COMMAND} -E cat
      # ${PROJECT_SOURCE_DIR}/testpy-output/*.command-line >
      # ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h
      COMMAND ${cat_command} ${PROJECT_SOURCE_DIR}/testpy-output/*.command-line
              > ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h 2> NULL
      DEPENDS run-introspected-command-line
    )

    add_custom_target(
      update_doxygen_version
      COMMAND ${PROJECT_SOURCE_DIR}/doc/ns3_html_theme/get_version.sh
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )

    add_custom_target(
      doxygen
      COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/doxygen.conf
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS update_doxygen_version run-print-introspected-doxygen
              assemble-introspected-command-line
      USES_TERMINAL
    )

    add_custom_target(
      doxygen-no-build
      COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/doxygen.conf
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS update_doxygen_version
      USES_TERMINAL
    )
  endif()

  # Now we check for sphinx dependencies
  mark_as_advanced(
    SPHINX_EXECUTABLE SPHINX_OUTPUT_HTML SPHINX_OUTPUT_MAN
    SPHINX_WARNINGS_AS_ERRORS
  )

  # Check deps accepts a list of packages, list of programs and name of the
  # return variable
  check_deps(
    "Sphinx" "epstopdf;pdflatex;latexmk;convert;dvipng"
    sphinx_docs_missing_deps
  )
  if(sphinx_docs_missing_deps)
    message(
      ${HIGHLIGHTED_STATUS}
      "docs: sphinx documentation not enabled due to missing dependencies: ${sphinx_docs_missing_deps}"
    )
    # cmake-format: off
    set(sphinx_missing_msg
        echo The following Sphinx dependencies are missing: ${sphinx_docs_missing_deps}.
            Reconfigure the project after installing them.
    )
    # cmake-format: on

    # Create stub targets to inform users about missing dependencies
    add_custom_target(sphinx COMMAND ${sphinx_missing_msg})
    add_custom_target(sphinx_manual COMMAND ${sphinx_missing_msg})
    add_custom_target(sphinx_models COMMAND ${sphinx_missing_msg})
    add_custom_target(sphinx_tutorial COMMAND ${sphinx_missing_msg})
    add_custom_target(sphinx_contributing COMMAND ${sphinx_missing_msg})
  else()
    add_custom_target(sphinx COMMENT "Building sphinx documents")

    function(sphinx_target targetname)
      # cmake-format: off
      add_custom_target(
        sphinx_${targetname}
        COMMAND make SPHINXOPTS=-N -k html singlehtml latexpdf
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doc/${targetname}
      )
      # cmake-format: on
      add_dependencies(sphinx sphinx_${targetname})
    endfunction()
    sphinx_target(manual)
    sphinx_target(models)
    sphinx_target(tutorial)
    sphinx_target(contributing)
  endif()
  # end of checking for documentation dependencies and creating targets

  # Adding this module manually is required by CMake 3.10
  include(CheckCXXSourceCompiles)

  # Process core-config If INT128 is not found, fallback to CAIRO
  if(${NS3_INT64X64} MATCHES "INT128")
    check_cxx_source_compiles(
      "#include <stdint.h>
       int main(int argc, char **argv)
         {
            (void)argc; (void)argv;
            if ((uint128_t *) 0) return 0;
            if (sizeof (uint128_t)) return 0;
            return 1;
         }"
      HAVE_UINT128_T
    )
    check_cxx_source_compiles(
      "#include <stdint.h>
       int main(int argc, char **argv)
         {
           (void)argc; (void)argv;
           if ((__uint128_t *) 0) return 0;
           if (sizeof (__uint128_t)) return 0;
           return 1;
        }"
      HAVE___UINT128_T
    )
    if(HAVE_UINT128_T OR HAVE___UINT128_T)
      set(INT64X64_USE_128 TRUE)
    else()
      message(${HIGHLIGHTED_STATUS}
              "Int128 was not found. Falling back to Cairo."
      )
      set(NS3_INT64X64 "CAIRO")
    endif()
  endif()

  # If long double and double have different sizes, fallback to CAIRO
  if(${NS3_INT64X64} MATCHES "DOUBLE")
    # WSLv1 has a long double issue that will result in a few tests failing
    # https://github.com/microsoft/WSL/issues/830
    include(CheckTypeSize)
    check_type_size("double" SIZEOF_DOUBLE)
    check_type_size("long double" SIZEOF_LONG_DOUBLE)

    if(${SIZEOF_LONG_DOUBLE} EQUAL ${SIZEOF_DOUBLE})
      message(
        STATUS
          "Long double has the wrong size: LD ${SIZEOF_LONG_DOUBLE} vs D ${SIZEOF_DOUBLE}. Falling back to CAIRO."
      )
      set(NS3_INT64X64 "CAIRO")
    else()
      set(INT64X64_USE_DOUBLE TRUE)
    endif()
  endif()

  # Fallback option
  if(${NS3_INT64X64} MATCHES "CAIRO")
    set(INT64X64_USE_CAIRO TRUE)
  endif()

  include(CheckIncludeFileCXX) # Used to check a single header at a time
  include(CheckIncludeFiles) # Used to check multiple headers at once
  include(CheckFunctionExists)

  # Check for required headers and functions, set flags if they're found or warn
  # if they're not found
  check_include_file_cxx("stdint.h" "HAVE_STDINT_H")
  check_include_file_cxx("inttypes.h" "HAVE_INTTYPES_H")
  check_include_file_cxx("sys/types.h" "HAVE_SYS_TYPES_H")
  check_include_file_cxx("sys/stat.h" "HAVE_SYS_STAT_H")
  check_include_file_cxx("dirent.h" "HAVE_DIRENT_H")
  check_include_file_cxx("stdlib.h" "HAVE_STDLIB_H")
  check_include_file_cxx("signal.h" "HAVE_SIGNAL_H")
  check_include_file_cxx("netpacket/packet.h" "HAVE_PACKETH")
  check_function_exists("getenv" "HAVE_GETENV")

  configure_file(
    build-support/core-config-template.h
    ${CMAKE_HEADER_OUTPUT_DIRECTORY}/core-config.h
  )

  # Force enable ns-3 logging in debug builds and if requested for other build
  # types
  if(${NS3_LOG} OR (${build_profile} STREQUAL "debug"))
    add_definitions(-DNS3_LOG_ENABLE)
  endif()
  # Force enable ns-3 asserts in debug builds and if requested for other build
  # types
  if(${NS3_ASSERT} OR (${build_profile} STREQUAL "debug"))
    add_definitions(-DNS3_ASSERT_ENABLE)
  endif()

  # Enable examples as tests suites
  if(${ENABLE_EXAMPLES})
    set(NS3_ENABLE_EXAMPLES "1")
    add_definitions(-DNS3_ENABLE_EXAMPLES -DCMAKE_EXAMPLE_AS_TEST)
  endif()

  set(ENABLE_TAP OFF)
  if(${NS3_TAP})
    set(ENABLE_TAP ON)
  endif()

  set(ENABLE_EMU OFF)
  if(${NS3_EMU})
    set(ENABLE_EMU ON)
  endif()

  set(PLATFORM_UNSUPPORTED_PRE "Platform doesn't support")
  set(PLATFORM_UNSUPPORTED_POST "features. Continuing without them.")
  # Remove from libs_to_build all incompatible libraries or the ones that
  # dependencies couldn't be installed
  if(APPLE OR WSLv1)
    set(ENABLE_TAP OFF)
    set(ENABLE_EMU OFF)
    list(REMOVE_ITEM libs_to_build fd-net-device)
    message(
      STATUS
        "${PLATFORM_UNSUPPORTED_PRE} TAP and EMU ${PLATFORM_UNSUPPORTED_POST}"
    )
  endif()

  if(NOT ${ENABLE_MPI})
    list(REMOVE_ITEM libs_to_build mpi)
  endif()

  if(NOT ${ENABLE_MTP})
    list(REMOVE_ITEM libs_to_build mtp)
  endif()

  if(NOT ${ENABLE_VISUALIZER})
    list(REMOVE_ITEM libs_to_build visualizer)
  endif()

  if(NOT ${ENABLE_TAP})
    list(REMOVE_ITEM libs_to_build tap-bridge)
  endif()

  # Create library names to solve dependency problems with macros that will be
  # called at each lib subdirectory
  set(ns3-libs)
  set(ns3-all-enabled-modules)
  set(ns3-libs-tests)
  set(ns3-contrib-libs)
  set(lib-ns3-static-objs)
  set(ns3-external-libs)
  set(ns3-python-bindings ns${NS3_VER}-pybindings${build_profile_suffix})
  set(ns3-python-bindings-modules)

  foreach(libname ${scanned_modules})
    # Create libname of output library of module
    library_target_name(${libname} targetname)
    mark_as_advanced(lib${libname} lib${libname}-obj)
    set(lib${libname} ${targetname} CACHE INTERNAL "")
    set(lib${libname}-obj ${targetname}-obj CACHE INTERNAL "")
  endforeach()

  unset(optional_visualizer_lib)
  if(${ENABLE_VISUALIZER} AND (visualizer IN_LIST libs_to_build))
    set(optional_visualizer_lib ${libvisualizer})
  endif()

  set(PRECOMPILE_HEADERS_ENABLED OFF)
  if(${NS3_PRECOMPILE_HEADERS})
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
      set(PRECOMPILE_HEADERS_ENABLED ON)
      message(STATUS "Precompiled headers were enabled")
    else()
      message(
        STATUS
          "CMake ${CMAKE_VERSION} does not support precompiled headers. Continuing without them"
      )
    endif()
  endif()

  if(${PRECOMPILE_HEADERS_ENABLED})
    set(precompiled_header_libraries
        <iostream>
        <stdint.h>
        <stdlib.h>
        <map>
        <unordered_map>
        <vector>
        <list>
        <algorithm>
        <string>
        <set>
        <sstream>
        <fstream>
        <cstdlib>
        <exception>
        <cstring>
        <limits>
        <math.h>
    )
    add_library(stdlib_pch OBJECT ${PROJECT_SOURCE_DIR}/build-support/empty.cc)
    target_precompile_headers(
      stdlib_pch PUBLIC "${precompiled_header_libraries}"
    )

    add_executable(
      stdlib_pch_exec ${PROJECT_SOURCE_DIR}/build-support/empty-main.cc
    )
    target_precompile_headers(
      stdlib_pch_exec PUBLIC "${precompiled_header_libraries}"
    )
    set_runtime_outputdirectory(stdlib_pch_exec ${CMAKE_BINARY_DIR}/ "")
  endif()

  # Create new lib for NS3 static builds
  set(lib-ns3-static ns${NS3_VER}-static${build_profile_suffix})

  # Create a new lib for NS3 monolibrary shared builds
  set(lib-ns3-monolib ns${NS3_VER}-monolib${build_profile_suffix})

  # All contrib libraries can be linked afterwards linking with
  # ${ns3-contrib-libs}
  process_contribution("${contrib_libs_to_build}")

  # Netanim depends on ns-3 core, so we built it later
  if(${NS3_NETANIM})
    include(FetchContent)
    FetchContent_Declare(
      netanim GIT_REPOSITORY https://gitlab.com/nsnam/netanim.git
      GIT_TAG netanim-3.108
    )
    FetchContent_Populate(netanim)
    file(COPY build-support/3rd-party/netanim-cmakelists.cmake
         DESTINATION ${netanim_SOURCE_DIR}
    )
    file(RENAME ${netanim_SOURCE_DIR}/netanim-cmakelists.cmake
         ${netanim_SOURCE_DIR}/CMakeLists.txt
    )
    add_subdirectory(${netanim_SOURCE_DIR})
  endif()
endmacro()

function(set_runtime_outputdirectory target_name output_directory target_prefix)
  set(ns3-exec-outputname ns${NS3_VER}-${target_name}${build_profile_suffix})
  set(ns3-execs "${output_directory}${ns3-exec-outputname};${ns3-execs}"
      CACHE INTERNAL "list of c++ executables"
  )

  set_target_properties(
    ${target_prefix}${target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_directory}
               RUNTIME_OUTPUT_NAME ${ns3-exec-outputname}
  )
  if(${XCODE})
    # Is that so hard not to break people's CI, AAPL?? Why would you output the
    # targets to a Debug/Release subfolder? Why?
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
      set_target_properties(
        ${target_prefix}${target_name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${output_directory}
                   RUNTIME_OUTPUT_NAME_${OUTPUTCONFIG} ${ns3-exec-outputname}
      )
    endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)
  endif()

  if(${ENABLE_TESTS})
    add_dependencies(all-test-targets ${target_prefix}${target_name})
    # Create a CTest entry for each executable
    if(WIN32)
      # Windows require this workaround to make sure the DLL files are located
      add_test(
        NAME ctest-${target_prefix}${target_name}
        COMMAND
          ${CMAKE_COMMAND} -E env
          "PATH=$ENV{PATH};${CMAKE_RUNTIME_OUTPUT_DIRECTORY};${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
          ${ns3-exec-outputname}
        WORKING_DIRECTORY ${output_directory}
      )
    else()
      add_test(NAME ctest-${target_prefix}${target_name}
               COMMAND ${ns3-exec-outputname}
               WORKING_DIRECTORY ${output_directory}
      )
    endif()
  endif()

  if(${NS3_CLANG_TIMETRACE})
    add_dependencies(timeTraceReport ${target_prefix}${target_name})
  endif()
endfunction(set_runtime_outputdirectory)

function(scan_python_examples path)
  # Skip python examples search in case the bindings are disabled
  if(NOT ${ENABLE_PYTHON_BINDINGS})
    return()
  endif()

  # Search for python examples
  file(GLOB_RECURSE python_examples ${path}/*.py)
  foreach(python_example ${python_examples})
    if(NOT (${python_example} MATCHES "examples-to-run"))
      set(ns3-execs-py "${python_example};${ns3-execs-py}"
          CACHE INTERNAL "list of python scripts"
      )
    endif()
  endforeach()
endfunction()

add_custom_target(copy_all_headers)
function(copy_headers_before_building_lib libname outputdir headers visibility)
  # cmake-format: off
  set(batch_symlinks)
  foreach(header ${headers})
    # Copy header to output directory on changes -> too darn slow
    # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${header} ${outputdir}/
    # COPYONLY)

    get_filename_component(
      header_name ${CMAKE_CURRENT_SOURCE_DIR}/${header} NAME
    )

    # If header already exists, skip symlinking/stub header creation
    if(EXISTS ${outputdir}/${header_name})
      continue()
    endif()

    # CMake 3.13 cannot create symlinks on Windows, so we use stub headers as a
    # fallback
    if(WIN32 AND (${CMAKE_VERSION} VERSION_LESS "3.13.0"))
      # Create a stub header in the output directory, including the real header
      # inside their respective module

      file(WRITE ${outputdir}/${header_name}
           "#include \"${CMAKE_CURRENT_SOURCE_DIR}/${header}\"\n"
      )
    else()
      # Create a symlink in the output directory to the original header Calling
      # execute_process for each symlink is too slow too, so we create a batch
      # with all headers execute_process(COMMAND ${CMAKE_COMMAND} -E
      # create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/${header}
      # ${outputdir}/${header_name})
      set(batch_symlinks
          ${batch_symlinks}
          COMMAND
          ${CMAKE_COMMAND}
          -E
          create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/${header}
          ${outputdir}/${header_name}
      )
    endif()
  endforeach()

  # Create all symlinks in a single call if there is a symlink to be done
  if(batch_symlinks)
    execute_process(${batch_symlinks})
  endif()
  # cmake-format: on
endfunction(copy_headers_before_building_lib)

function(remove_lib_prefix prefixed_library library)
  # Check if there is a lib prefix
  string(FIND "${prefixed_library}" "lib" lib_pos)

  # If there is a lib prefix, try to remove it
  if(${lib_pos} EQUAL 0)
    # Check if we still have something remaining after removing the "lib" prefix
    string(LENGTH ${prefixed_library} len)
    if(${len} LESS 4)
      message(FATAL_ERROR "Invalid library name: ${prefixed_library}")
    endif()

    # Remove lib prefix from module name (e.g. libcore -> core)
    string(SUBSTRING "${prefixed_library}" 3 -1 unprefixed_library)
  else()
    set(unprefixed_library ${prefixed_library})
  endif()

  # Save the unprefixed library name to the parent scope
  set(${library} ${unprefixed_library} PARENT_SCOPE)
endfunction()

function(check_for_missing_libraries output_variable_name libraries)
  set(missing_dependencies)
  foreach(lib ${libraries})
    # skip check for ns-3 modules if its a path to a library
    if(EXISTS ${lib})
      continue()
    endif()

    # check if the example depends on disabled modules
    remove_lib_prefix("${lib}" lib)

    # Check if the module exists in the ns-3 modules list or if it is a
    # 3rd-party library
    if(NOT (${lib} IN_LIST ns3-all-enabled-modules))
      list(APPEND missing_dependencies ${lib})
    endif()
  endforeach()
  set(${output_variable_name} ${missing_dependencies} PARENT_SCOPE)
endfunction()

# Import macros used for modules and define specialized versions for src modules
include(build-support/custom-modules/ns3-module-macros.cmake)

# Contrib modules counterparts of macros above
include(build-support/custom-modules/ns3-contributions.cmake)

# Macro to build examples in ns-3-dev/examples/
macro(build_example)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SOURCE_FILES HEADER_FILES LIBRARIES_TO_LINK)
  cmake_parse_arguments(
    "EXAMPLE" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  check_for_missing_libraries(
    missing_dependencies "${EXAMPLE_LIBRARIES_TO_LINK}"
  )

  if(NOT missing_dependencies)
    # Create shared library with sources and headers
    add_executable(
      ${EXAMPLE_NAME} "${EXAMPLE_SOURCE_FILES}" "${EXAMPLE_HEADER_FILES}"
    )

    if(${NS3_STATIC})
      target_link_libraries(
        ${EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE_STATIC} ${lib-ns3-static}
      )
    elseif(${NS3_MONOLIB})
      target_link_libraries(
        ${EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE} ${lib-ns3-monolib}
        ${LIB_AS_NEEDED_POST}
      )
    else()
      # Link the shared library with the libraries passed
      target_link_libraries(
        ${EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE} ${EXAMPLE_LIBRARIES_TO_LINK}
        ${optional_visualizer_lib} ${LIB_AS_NEEDED_POST}
      )
    endif()

    if(${PRECOMPILE_HEADERS_ENABLED})
      target_precompile_headers(${EXAMPLE_NAME} REUSE_FROM stdlib_pch_exec)
    endif()

    set_runtime_outputdirectory(
      ${EXAMPLE_NAME}
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/${examplefolder}/ ""
    )
  endif()
endmacro()

function(filter_modules modules_to_filter all_modules_list filter_in)
  set(new_modules_to_build)
  # We are receiving variable names as arguments, so we have to "dereference"
  # them first That is why we have the duplicated ${${}}
  foreach(module ${${all_modules_list}})
    if(${filter_in} (${module} IN_LIST ${modules_to_filter}))
      list(APPEND new_modules_to_build ${module})
    endif()
  endforeach()
  set(${all_modules_list} ${new_modules_to_build} PARENT_SCOPE)
endfunction()

function(resolve_dependencies module_name dependencies contrib_dependencies)
  # Create cache variables to hold dependencies list and visited
  set(dependency_visited "" CACHE INTERNAL "")
  set(contrib_dependency_visited "" CACHE INTERNAL "")
  recursive_dependency(${module_name})
  if(${module_name} IN_LIST dependency_visited)
    set(temp ${dependency_visited})
    list(REMOVE_AT temp 0)
    set(${dependencies} ${temp} PARENT_SCOPE)
    set(${contrib_dependencies} ${contrib_dependency_visited} PARENT_SCOPE)
  else()
    set(temp ${contrib_dependency_visited})
    list(REMOVE_AT temp 0)
    set(${dependencies} ${dependency_visited} PARENT_SCOPE)
    set(${contrib_dependencies} ${temp} PARENT_SCOPE)
  endif()
  unset(dependency_visited CACHE)
  unset(contrib_dependency_visited CACHE)
endfunction()

function(filter_libraries cmakelists_contents libraries)
  string(REGEX MATCHALL "{lib[^}]*[^obj]}" matches "${cmakelists_content}")
  list(REMOVE_ITEM matches "{libraries_to_link}")
  string(REPLACE "{lib\${name" "" matches "${matches}") # special case for
                                                        # src/test
  string(REPLACE "{lib" "" matches "${matches}")
  string(REPLACE "}" "" matches "${matches}")
  set(${libraries} ${matches} PARENT_SCOPE)
endfunction()

function(recursive_dependency module_name)
  # Read module CMakeLists.txt and search for ns-3 modules
  set(src_cmakelist ${PROJECT_SOURCE_DIR}/src/${module_name}/CMakeLists.txt)
  set(contrib_cmakelist
      ${PROJECT_SOURCE_DIR}/contrib/${module_name}/CMakeLists.txt
  )
  set(contrib FALSE)
  if(EXISTS ${src_cmakelist})
    file(READ ${src_cmakelist} cmakelists_content)
  elseif(EXISTS ${contrib_cmakelist})
    file(READ ${contrib_cmakelist} cmakelists_content)
    set(contrib TRUE)
  else()
    set(cmakelists_content "")
    message(${HIGHLIGHTED_STATUS}
            "The CMakeLists.txt file for module ${module_name} was not found."
    )
  endif()

  filter_libraries("${cmakelists_content}" matches)

  # Add this visited module dependencies to the dependencies list
  if(contrib)
    set(contrib_dependency_visited
        "${contrib_dependency_visited};${module_name}" CACHE INTERNAL ""
    )
    set(examples_cmakelists ${contrib_cmakelist})
  else()
    set(dependency_visited "${dependency_visited};${module_name}" CACHE INTERNAL
                                                                        ""
    )
    set(examples_cmakelists ${src_cmakelist})
  endif()

  # cmake-format: off
  # Scan dependencies required by this module examples
  #if(${ENABLE_EXAMPLES})
  #  string(REPLACE "${module_name}" "${module_name}/examples" examples_cmakelists ${examples_cmakelists})
  #  if(EXISTS ${examples_cmakelists})
  #    file(READ ${examples_cmakelists} cmakelists_content)
  #    filter_libraries(${cmakelists_content} example_matches)
  #  endif()
  #endif()
  # cmake-format: on

  # For each dependency, call this same function
  set(matches "${matches};${example_matches}")
  foreach(match ${matches})
    if(NOT ((${match} IN_LIST dependency_visited)
            OR (${match} IN_LIST contrib_dependency_visited))
    )
      recursive_dependency(${match})
    endif()
  endforeach()
endfunction()

macro(filter_enabled_and_disabled_modules libs_to_build contrib_libs_to_build
      NS3_ENABLED_MODULES NS3_DISABLED_MODULES ns3rc_enabled_modules
)
  mark_as_advanced(ns3-all-enabled-modules)

  # Before filtering, we set a variable with all scanned modules in the src
  # directory
  set(scanned_modules ${${libs_to_build}})

  # Ensure enabled and disable modules lists are using semicolons
  string(REPLACE "," ";" ${NS3_ENABLED_MODULES} "${${NS3_ENABLED_MODULES}}")
  string(REPLACE "," ";" ${NS3_DISABLED_MODULES} "${${NS3_DISABLED_MODULES}}")

  # Now that scanning modules finished, we can remove the disabled modules or
  # replace the modules list with the ones in the enabled list
  if(${NS3_ENABLED_MODULES} OR ${ns3rc_enabled_modules})
    # List of enabled modules passed by the command line overwrites list read
    # from ns3rc
    if(${NS3_ENABLED_MODULES})
      set(ns3rc_enabled_modules ${${NS3_ENABLED_MODULES}})
    endif()

    # Filter enabled modules
    filter_modules(ns3rc_enabled_modules libs_to_build "")
    filter_modules(ns3rc_enabled_modules contrib_libs_to_build "")

    # Use recursion to automatically determine dependencies required by the
    # manually enabled modules
    foreach(lib ${${contrib_libs_to_build}})
      resolve_dependencies(${lib} dependencies contrib_dependencies)
      list(APPEND ${contrib_libs_to_build} "${contrib_dependencies}")
      list(APPEND ${libs_to_build} "${dependencies}")
      unset(dependencies)
      unset(contrib_dependencies)
    endforeach()

    foreach(lib ${${libs_to_build}})
      resolve_dependencies(${lib} dependencies contrib_dependencies)
      list(APPEND ${libs_to_build} "${dependencies}")
      unset(dependencies)
      unset(contrib_dependencies)
    endforeach()

    if(core IN_LIST ${libs_to_build})
      list(APPEND ${libs_to_build} test) # include test module
    endif()
  endif()

  if(${NS3_DISABLED_MODULES})
    set(all_libs ${${libs_to_build}};${${contrib_libs_to_build}})

    # We then use the recursive dependency finding to get all dependencies of
    # all modules
    foreach(lib ${all_libs})
      resolve_dependencies(${lib} dependencies contrib_dependencies)
      set(${lib}_dependencies "${dependencies};${contrib_dependencies}")
      unset(dependencies)
      unset(contrib_dependencies)
    endforeach()

    # Now we can begin removing libraries that require disabled dependencies
    set(disabled_libs "${${NS3_DISABLED_MODULES}}")
    foreach(libo ${all_libs})
      foreach(lib ${all_libs})
        foreach(disabled_lib ${disabled_libs})
          if(${lib} STREQUAL ${disabled_lib})
            continue()
          endif()
          if(${disabled_lib} IN_LIST ${lib}_dependencies)
            list(APPEND disabled_libs ${lib})
            break() # proceed to the next lib in all_libs
          endif()
        endforeach()
      endforeach()
    endforeach()

    # Clean dependencies lists
    foreach(lib ${all_libs})
      unset(${lib}_dependencies)
    endforeach()

    # We can filter out disabled modules
    filter_modules(disabled_libs libs_to_build "NOT")
    filter_modules(disabled_libs contrib_libs_to_build "NOT")

    if(core IN_LIST ${libs_to_build})
      list(APPEND ${libs_to_build} test) # include test module
    endif()
  endif()

  # Older CMake versions require this workaround for empty lists
  if(NOT ${contrib_libs_to_build})
    set(${contrib_libs_to_build} "")
  endif()

  # Filter out any eventual duplicates
  list(REMOVE_DUPLICATES ${libs_to_build})
  list(REMOVE_DUPLICATES ${contrib_libs_to_build})

  # Export list with all enabled modules (used to separate ns libraries from
  # non-ns libraries in ns3_module_macros)
  set(ns3-all-enabled-modules "${${libs_to_build}};${${contrib_libs_to_build}}"
      CACHE INTERNAL "list with all enabled modules"
  )
endmacro()

# Parse .ns3rc
function(parse_ns3rc enabled_modules examples_enabled tests_enabled)
  set(ns3rc ${PROJECT_SOURCE_DIR}/.ns3rc)
  # Set parent scope variables with default values (all modules, no examples nor
  # tests)
  set(${enabled_modules} "" PARENT_SCOPE)
  set(${examples_enabled} "FALSE" PARENT_SCOPE)
  set(${tests_enabled} "FALSE" PARENT_SCOPE)
  if(EXISTS ${ns3rc})
    # If ns3rc exists in ns-3-dev, read it
    file(READ ${ns3rc} ns3rc_contents)

    # Match modules_enabled list
    if(ns3rc_contents MATCHES "modules_enabled.*\\[(.*).*\\]")
      set(${enabled_modules} ${CMAKE_MATCH_1})
      if(${enabled_modules} MATCHES "all_modules")
        # If all modules, just clean the filter and all modules will get built
        # by default
        set(${enabled_modules})
      else()
        # If modules are listed, remove quotes and replace commas with
        # semicolons transforming a string into a cmake list
        string(REPLACE "," ";" ${enabled_modules} "${${enabled_modules}}")
        string(REPLACE "'" "" ${enabled_modules} "${${enabled_modules}}")
        string(REPLACE "\"" "" ${enabled_modules} "${${enabled_modules}}")
        string(REPLACE " " "" ${enabled_modules} "${${enabled_modules}}")
        string(REPLACE "\n" ";" ${enabled_modules} "${${enabled_modules}}")
        list(SORT ${enabled_modules})

        # Remove possibly empty entry
        list(REMOVE_ITEM ${enabled_modules} "")
        foreach(element ${${enabled_modules}})
          # Inspect each element for comments
          if(${element} MATCHES "#.*")
            list(REMOVE_ITEM ${enabled_modules} ${element})
          endif()
        endforeach()
      endif()
    endif()

    # Match examples_enabled flag
    if(ns3rc_contents MATCHES "examples_enabled = (True|False)")
      set(${examples_enabled} ${CMAKE_MATCH_1})
    endif()

    # Match tests_enabled flag
    if(ns3rc_contents MATCHES "tests_enabled = (True|False)")
      set(${tests_enabled} ${CMAKE_MATCH_1})
    endif()

    # Save variables to parent scope
    set(${enabled_modules} "${${enabled_modules}}" PARENT_SCOPE)
    set(${examples_enabled} "${${examples_enabled}}" PARENT_SCOPE)
    set(${tests_enabled} "${${tests_enabled}}" PARENT_SCOPE)
  endif()
endfunction(parse_ns3rc)

function(log_find_searched_paths)
  # Parse arguments
  set(options)
  set(oneValueArgs TARGET_TYPE TARGET_NAME SEARCH_RESULT SEARCH_SYSTEM_PREFIX)
  set(multiValueArgs SEARCH_PATHS SEARCH_SUFFIXES)
  cmake_parse_arguments(
    "LOGFIND" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  # Get searched paths and add cmake_system_prefix_path if not explicitly marked
  # not to include it
  set(tsearch_paths ${LOGFIND_SEARCH_PATHS})
  if("${LOGFIND_SEARCH_SYSTEM_PREFIX}" STREQUAL "")
    list(APPEND tsearch_paths "${CMAKE_SYSTEM_PREFIX_PATH}")
  endif()

  set(log_find
      "Looking for ${LOGFIND_TARGET_TYPE} ${LOGFIND_TARGET_NAME} in:\n"
  )
  # For each search path and suffix combination, print a line
  foreach(tsearch_path ${tsearch_paths})
    foreach(suffix ${LOGFIND_SEARCH_SUFFIXES})
      string(APPEND log_find
             "\t${tsearch_path}${suffix}/${LOGFIND_TARGET_NAME}\n"
      )
    endforeach()
  endforeach()

  # Add a final line saying if the file was found and where, or if it was not
  # found
  if("${${LOGFIND_SEARCH_RESULT}}" STREQUAL "${LOGFIND_SEARCH_RESULT}-NOTFOUND")
    string(APPEND log_find
           "\n\t${LOGFIND_TARGET_TYPE} ${LOGFIND_TARGET_NAME} was not found\n"
    )
  else()
    string(
      APPEND
      log_find
      "\n\t${LOGFIND_TARGET_TYPE} ${LOGFIND_TARGET_NAME} was found in ${${LOGFIND_SEARCH_RESULT}}\n"
    )
  endif()

  # Replace duplicate path separators
  string(REPLACE "//" "/" log_find "${log_find}")

  # Print find debug message similar to the one produced by
  # CMAKE_FIND_DEBUG_MODE=true in CMake >= 3.17
  message(STATUS ${log_find})
endfunction()

function(find_external_library)
  # Parse arguments
  set(options QUIET)
  set(oneValueArgs DEPENDENCY_NAME HEADER_NAME LIBRARY_NAME)
  set(multiValueArgs HEADER_NAMES LIBRARY_NAMES PATH_SUFFIXES SEARCH_PATHS)
  cmake_parse_arguments(
    "FIND_LIB" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  # Set the external package/dependency name
  set(name ${FIND_LIB_DEPENDENCY_NAME})

  # We process individual and list of headers and libraries by transforming them
  # into lists
  set(library_names "${FIND_LIB_LIBRARY_NAME};${FIND_LIB_LIBRARY_NAMES}")
  set(header_names "${FIND_LIB_HEADER_NAME};${FIND_LIB_HEADER_NAMES}")

  # Just changing the parsed argument name back to something shorter
  set(search_paths ${FIND_LIB_SEARCH_PATHS})
  set(path_suffixes "${FIND_LIB_PATH_SUFFIXES}")

  set(not_found_libraries)
  set(library_dirs)
  set(libraries)

  # Paths and suffixes where libraries will be searched on
  set(library_search_paths
      ${search_paths}
      ${CMAKE_OUTPUT_DIRECTORY} # Search for libraries in ns-3-dev/build
      ${CMAKE_INSTALL_PREFIX} # Search for libraries in the install directory
                              # (e.g. /usr/)
      $ENV{LD_LIBRARY_PATH} # Search for libraries in LD_LIBRARY_PATH
                            # directories
      $ENV{PATH} # Search for libraries in PATH directories
  )
  set(suffixes /build /lib /build/lib / /bin ${path_suffixes})

  # For each of the library names in LIBRARY_NAMES or LIBRARY_NAME
  foreach(library ${library_names})
    # We mark this value is advanced not to pollute the configuration with
    # ccmake with the cache variables used internally
    mark_as_advanced(${name}_library_internal_${library})

    # We search for the library named ${library} and store the results in
    # ${name}_library_internal_${library}
    find_library(
      ${name}_library_internal_${library} ${library}
      HINTS ${library_search_paths} PATH_SUFFIXES ${suffixes}
    )
    # cmake-format: off
    # Note: the PATH_SUFFIXES above apply to *ALL* PATHS and HINTS Which
    # translates to CMake searching on standard library directories
    # CMAKE_SYSTEM_PREFIX_PATH, user-settable CMAKE_PREFIX_PATH or
    # CMAKE_LIBRARY_PATH and the directories listed above
    #
    # e.g.  from Ubuntu 22.04 CMAKE_SYSTEM_PREFIX_PATH =
    # /usr/local;/usr;/;/usr/local;/usr/X11R6;/usr/pkg;/opt
    #
    # Searched directories without suffixes
    #
    # ${CMAKE_SYSTEM_PREFIX_PATH}[0] = /usr/local/
    # ${CMAKE_SYSTEM_PREFIX_PATH}[1] = /usr
    # ${CMAKE_SYSTEM_PREFIX_PATH}[2] = /
    # ...
    # ${CMAKE_SYSTEM_PREFIX_PATH}[6] = /opt
    # ${LD_LIBRARY_PATH}[0]
    # ...
    # ${LD_LIBRARY_PATH}[m]
    # ${PATH}[0]
    # ...
    # ${PATH}[m]
    #
    # Searched directories with suffixes include all of the directories above
    # plus all suffixes
    # PATH_SUFFIXES /build /lib /build/lib / /bin # ${path_suffixes}
    #
    # /usr/local/build
    # /usr/local/lib
    # /usr/local/build/lib
    # /usr/local/bin
    # ...
    #
    # cmake-format: on
    # Or enable NS3_VERBOSE to print the searched paths

    # Print tested paths to the searched library and if it was found
    if(${NS3_VERBOSE} AND (${CMAKE_VERSION} VERSION_LESS "3.17.0"))
      log_find_searched_paths(
        TARGET_TYPE
        Library
        TARGET_NAME
        ${library}
        SEARCH_RESULT
        ${name}_library_internal_${library}
        SEARCH_PATHS
        ${library_search_paths}
        SEARCH_SUFFIXES
        ${suffixes}
      )
    endif()

    # After searching the library, the internal variable should have either the
    # absolute path to the library or the name of the variable appended with
    # -NOTFOUND
    if("${${name}_library_internal_${library}}" STREQUAL
       "${name}_library_internal_${library}-NOTFOUND"
    )
      # We keep track of libraries that were not found
      list(APPEND not_found_libraries ${library})
    else()
      # We get the name of the parent directory of the library and append the
      # library to a list of found libraries
      get_filename_component(
        ${name}_library_dir_internal ${${name}_library_internal_${library}}
        DIRECTORY
      ) # e.g. lib/openflow.(so|dll|dylib|a) -> lib
      list(APPEND library_dirs ${${name}_library_dir_internal})
      list(APPEND libraries ${${name}_library_internal_${library}})
    endif()
  endforeach()

  # For each library that was found (e.g. /usr/lib/pthread.so), get their parent
  # directory (/usr/lib) and its parent (/usr)
  set(parent_dirs)
  foreach(libdir ${library_dirs})
    get_filename_component(parent_libdir ${libdir} DIRECTORY)
    get_filename_component(parent_parent_libdir ${parent_libdir} DIRECTORY)
    list(APPEND parent_dirs ${libdir} ${parent_libdir} ${parent_parent_libdir})
  endforeach()

  # If we already found a library somewhere, limit the search paths for the
  # header
  if(parent_dirs)
    set(header_search_paths ${parent_dirs})
    set(header_skip_system_prefix NO_CMAKE_SYSTEM_PATH)
  else()
    set(header_search_paths
        ${search_paths} ${CMAKE_OUTPUT_DIRECTORY} # Search for headers in
                                                  # ns-3-dev/build
        ${CMAKE_INSTALL_PREFIX} # Search for headers in the install
    )
  endif()

  set(not_found_headers)
  set(include_dirs)
  foreach(header ${header_names})
    # The same way with libraries, we mark the internal variable as advanced not
    # to pollute ccmake configuration with variables used internally
    mark_as_advanced(${name}_header_internal_${header})
    set(suffixes
        /build
        /include
        /build/include
        /build/include/${name}
        /include/${name}
        /${name}
        /
        ${path_suffixes}
    )

    # cmake-format: off
    # Here we search for the header file named ${header} and store the result in
    # ${name}_header_internal_${header}
    #
    # The same way we did with libraries, here we search on
    # CMAKE_SYSTEM_PREFIX_PATH, along with user-settable ${search_paths}, the
    # parent directories from the libraries, CMAKE_OUTPUT_DIRECTORY and
    # CMAKE_INSTALL_PREFIX
    #
    # And again, for each of them, for every suffix listed /usr/local/build
    # /usr/local/include
    # /usr/local/build/include
    # /usr/local/build/include/${name}
    # /usr/local/include/${name}
    # ...
    #
    # cmake-format: on
    # Or enable NS3_VERBOSE to get the searched paths printed while configuring

    find_file(${name}_header_internal_${header} ${header}
              HINTS ${header_search_paths} # directory (e.g. /usr/)
                    ${header_skip_system_prefix} PATH_SUFFIXES ${suffixes}
    )

    # Print tested paths to the searched header and if it was found
    if(${NS3_VERBOSE} AND (${CMAKE_VERSION} VERSION_LESS "3.17.0"))
      log_find_searched_paths(
        TARGET_TYPE
        Header
        TARGET_NAME
        ${header}
        SEARCH_RESULT
        ${name}_header_internal_${header}
        SEARCH_PATHS
        ${header_search_paths}
        SEARCH_SUFFIXES
        ${suffixes}
        SEARCH_SYSTEM_PREFIX
        ${header_skip_system_prefix}
      )
    endif()

    # If the header file was not found, append to the not-found list
    if("${${name}_header_internal_${header}}" STREQUAL
       "${name}_header_internal_${header}-NOTFOUND"
    )
      list(APPEND not_found_headers ${header})
    else()
      # If the header file was found, get their directories and the parent of
      # their directories to add as include directories
      get_filename_component(
        header_include_dir ${${name}_header_internal_${header}} DIRECTORY
      ) # e.g. include/click/ (simclick.h) -> #include <simclick.h> should work
      get_filename_component(
        header_include_dir2 ${header_include_dir} DIRECTORY
      ) # e.g. include/(click) -> #include <click/simclick.h> should work
      list(APPEND include_dirs ${header_include_dir} ${header_include_dir2})
    endif()
  endforeach()

  # Remove duplicate include directories
  if(include_dirs)
    list(REMOVE_DUPLICATES include_dirs)
  endif()

  # If we find both library and header, we export their values
  if((NOT not_found_libraries) AND (NOT not_found_headers))
    set(${name}_INCLUDE_DIRS "${include_dirs}" PARENT_SCOPE)
    set(${name}_LIBRARIES "${libraries}" PARENT_SCOPE)
    set(${name}_HEADER ${${name}_header_internal} PARENT_SCOPE)
    set(${name}_FOUND TRUE PARENT_SCOPE)
    if(NOT ${FIND_LIB_QUIET})
      message(STATUS "find_external_library: ${name} was found.")
    endif()
  else()
    set(${name}_INCLUDE_DIRS PARENT_SCOPE)
    set(${name}_LIBRARIES PARENT_SCOPE)
    set(${name}_HEADER PARENT_SCOPE)
    set(${name}_FOUND FALSE PARENT_SCOPE)
    if(NOT ${FIND_LIB_QUIET})
      message(
        ${HIGHLIGHTED_STATUS}
        "find_external_library: ${name} was not found. Missing headers: \"${not_found_headers}\" and missing libraries: \"${not_found_libraries}\"."
      )
    endif()
  endif()
endfunction()

function(write_header_to_modules_map)
  if(${ENABLE_SCAN_PYTHON_BINDINGS})
    set(header_map ${ns3-headers-to-module-map})

    # Trim last comma
    string(LENGTH "${header_map}" header_map_len)
    math(EXPR header_map_len "${header_map_len}-1")
    string(SUBSTRING "${header_map}" 0 ${header_map_len} header_map)

    # Then write to header_map.json for consumption of pybindgen
    file(WRITE ${PROJECT_BINARY_DIR}/header_map.json "{${header_map}}")
  endif()
endfunction()

function(get_target_includes target output)
  set(include_directories)
  get_target_property(include_dirs ${target} INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES include_dirs)
  foreach(include_dir ${include_dirs})
    if(include_dir MATCHES "<")
      # Skip CMake build and install interface includes
      continue()
    else()
      # Append the include directory to a list
      set(include_directories ${include_directories} -I${include_dir})
    endif()
  endforeach()
  set(${output} ${include_directories} PARENT_SCOPE)
endfunction()

function(check_python_packages packages missing_packages)
  set(missing)
  foreach(package ${packages})
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import ${package}"
      RESULT_VARIABLE return_code OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT (${return_code} EQUAL 0))
      list(APPEND missing ${package})
    endif()
  endforeach()
  set(${missing_packages} "${missing}" PARENT_SCOPE)
endfunction()

include(build-support/custom-modules/ns3-lock.cmake)
include(build-support/custom-modules/ns3-configtable.cmake)
