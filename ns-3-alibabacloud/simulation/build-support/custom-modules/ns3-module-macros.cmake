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

# cmake-format: off
#
# This macro processes a ns-3 module
#
# Arguments:
# LIBNAME = core, wifi, contributor_module
# SOURCE_FILES = "list;of;.cc;files;"
# HEADER_FILES = "list;of;public;.h;files;"
# LIBRARIES_TO_LINK = "list;of;${library_names};"
# TEST_SOURCES = "list;of;.cc;test;files;"
#
# DEPRECATED_HEADER_FILES = "list;of;deprecated;.h;files", copy won't get triggered if DEPRECATED_HEADER_FILES isn't set
# IGNORE_PCH = TRUE or FALSE, prevents the PCH from including undesired system libraries (e.g. custom GLIBC for DCE)
# MODULE_ENABLED_FEATURES = "list;of;enabled;features;for;this;module" (used by fd-net-device)
# cmake-format: on

function(build_lib)
  # Argument parsing
  set(options IGNORE_PCH)
  set(oneValueArgs LIBNAME)
  set(multiValueArgs SOURCE_FILES HEADER_FILES LIBRARIES_TO_LINK TEST_SOURCES
                     DEPRECATED_HEADER_FILES MODULE_ENABLED_FEATURES
  )
  cmake_parse_arguments(
    "BLIB" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  # Get path src/module or contrib/module
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" FOLDER
                 "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  # Add library to a global list of libraries
  if("${FOLDER}" MATCHES "src")
    set(ns3-libs "${lib${BLIB_LIBNAME}};${ns3-libs}"
        CACHE INTERNAL "list of processed upstream modules"
    )
  else()
    set(ns3-contrib-libs "${lib${BLIB_LIBNAME}};${ns3-contrib-libs}"
        CACHE INTERNAL "list of processed contrib modules"
    )
  endif()

  if(NOT ${XCODE})
    # Create object library with sources and headers, that will be used in
    # lib-ns3-static and the shared library
    add_library(
      ${lib${BLIB_LIBNAME}-obj} OBJECT "${BLIB_SOURCE_FILES}"
                                       "${BLIB_HEADER_FILES}"
    )

    if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${IGNORE_PCH}))
      target_precompile_headers(${lib${BLIB_LIBNAME}-obj} REUSE_FROM stdlib_pch)
    endif()

    # Create shared library with previously created object library (saving
    # compilation time for static libraries)
    add_library(
      ${lib${BLIB_LIBNAME}} SHARED $<TARGET_OBJECTS:${lib${BLIB_LIBNAME}-obj}>
    )
  else()
    # Xcode and CMake don't play well when using object libraries, so we have a
    # specific path for that
    add_library(${lib${BLIB_LIBNAME}} SHARED "${BLIB_SOURCE_FILES}")

    if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${IGNORE_PCH}))
      target_precompile_headers(${lib${BLIB_LIBNAME}} REUSE_FROM stdlib_pch)
    endif()
  endif()

  add_library(ns3::${lib${BLIB_LIBNAME}} ALIAS ${lib${BLIB_LIBNAME}})

  # Associate public headers with library for installation purposes
  if("${BLIB_LIBNAME}" STREQUAL "core")
    set(config_headers ${CMAKE_HEADER_OUTPUT_DIRECTORY}/config-store-config.h
                       ${CMAKE_HEADER_OUTPUT_DIRECTORY}/core-config.h
    )
    if(${ENABLE_BUILD_VERSION})
      list(APPEND config_headers
           ${CMAKE_HEADER_OUTPUT_DIRECTORY}/version-defines.h
      )
    endif()
  endif()
  set_target_properties(
    ${lib${BLIB_LIBNAME}}
    PROPERTIES
      PUBLIC_HEADER
      "${BLIB_HEADER_FILES};${BLIB_DEPRECATED_HEADER_FILES};${config_headers};${CMAKE_HEADER_OUTPUT_DIRECTORY}/${BLIB_LIBNAME}-module.h"
  )

  if(${NS3_CLANG_TIMETRACE})
    add_dependencies(timeTraceReport ${lib${BLIB_LIBNAME}})
  endif()

  # Split ns and non-ns libraries to manage their propagation properly
  set(non_ns_libraries_to_link ${CMAKE_THREAD_LIBS_INIT})

  set(ns_libraries_to_link)

  foreach(library ${BLIB_LIBRARIES_TO_LINK})
    remove_lib_prefix("${library}" module_name)

    # Check if the module exists in the ns-3 modules list or if it is a
    # 3rd-party library
    if(${module_name} IN_LIST ns3-all-enabled-modules)
      list(APPEND ns_libraries_to_link ${library})
    else()
      list(APPEND non_ns_libraries_to_link ${library})
    endif()
    unset(module_name)
  endforeach()

  if(NOT ${NS3_REEXPORT_THIRD_PARTY_LIBRARIES})
    # ns-3 libraries are linked publicly, to make sure other modules can find
    # each other without being directly linked
    set(exported_libraries PUBLIC ${LIB_AS_NEEDED_PRE} ${ns_libraries_to_link}
                           ${LIB_AS_NEEDED_POST}
    )

    # non-ns-3 libraries are linked privately, not propagating unnecessary
    # libraries such as pthread, librt, etc
    set(private_libraries PRIVATE ${LIB_AS_NEEDED_PRE}
                          ${non_ns_libraries_to_link} ${LIB_AS_NEEDED_POST}
    )

    # we don't re-export included libraries from 3rd-party modules
    set(exported_include_directories)
  else()
    # we export everything by default when NS3_REEXPORT_THIRD_PARTY_LIBRARIES=ON
    set(exported_libraries PUBLIC ${LIB_AS_NEEDED_PRE} ${ns_libraries_to_link}
                           ${non_ns_libraries_to_link} ${LIB_AS_NEEDED_POST}
    )
    set(private_libraries)

    # with NS3_REEXPORT_THIRD_PARTY_LIBRARIES, we export all 3rd-party library
    # include directories, allowing consumers of this module to include and link
    # the 3rd-party code with no additional setup
    get_target_includes(${lib${BLIB_LIBNAME}} exported_include_directories)
    string(REPLACE "-I" "" exported_include_directories
                   "${exported_include_directories}"
    )
    string(REPLACE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/include" ""
                   exported_include_directories
                   "${exported_include_directories}"
    )
  endif()

  target_link_libraries(
    ${lib${BLIB_LIBNAME}} ${exported_libraries} ${private_libraries}
  )

  # set output name of library
  set_target_properties(
    ${lib${BLIB_LIBNAME}}
    PROPERTIES OUTPUT_NAME ns${NS3_VER}-${BLIB_LIBNAME}${build_profile_suffix}
  )

  # export include directories used by this library so that it can be used by
  # 3rd-party consumers of ns-3 using find_package(ns3) this will automatically
  # add the build/include path to them, so that they can ns-3 headers with
  # <ns3/something.h>
  target_include_directories(
    ${lib${BLIB_LIBNAME}}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_OUTPUT_DIRECTORY}/include>
           $<INSTALL_INTERFACE:include>
    INTERFACE ${exported_include_directories}
  )

  set(ns3-external-libs "${non_ns_libraries_to_link};${ns3-external-libs}"
      CACHE INTERNAL
            "list of non-ns libraries to link to NS3_STATIC and NS3_MONOLIB"
  )
  if(${NS3_STATIC} OR ${NS3_MONOLIB})
    set(lib-ns3-static-objs
        "$<TARGET_OBJECTS:${lib${BLIB_LIBNAME}-obj}>;${lib-ns3-static-objs}"
        CACHE
          INTERNAL
          "list of object files from module used by NS3_STATIC and NS3_MONOLIB"
    )
  endif()

  # Write a module header that includes all headers from that module
  write_module_header("${BLIB_LIBNAME}" "${BLIB_HEADER_FILES}")

  # Copy all header files to outputfolder/include before each build
  copy_headers_before_building_lib(
    ${BLIB_LIBNAME} ${CMAKE_HEADER_OUTPUT_DIRECTORY} "${BLIB_HEADER_FILES}"
    public
  )
  if(BLIB_DEPRECATED_HEADER_FILES)
    copy_headers_before_building_lib(
      ${BLIB_LIBNAME} ${CMAKE_HEADER_OUTPUT_DIRECTORY}
      "${BLIB_DEPRECATED_HEADER_FILES}" deprecated
    )
  endif()

  # Build tests if requested
  if(${ENABLE_TESTS})
    list(LENGTH BLIB_TEST_SOURCES test_source_len)
    if(${test_source_len} GREATER 0)
      # Create BLIB_LIBNAME of output library test of module
      set(test${BLIB_LIBNAME} lib${BLIB_LIBNAME}-test CACHE INTERNAL "")
      set(ns3-libs-tests "${test${BLIB_LIBNAME}};${ns3-libs-tests}"
          CACHE INTERNAL "list of test libraries"
      )

      # Create shared library containing tests of the module
      add_library(${test${BLIB_LIBNAME}} SHARED "${BLIB_TEST_SOURCES}")

      # Link test library to the module library
      if(${NS3_MONOLIB})
        target_link_libraries(
          ${test${BLIB_LIBNAME}} ${LIB_AS_NEEDED_PRE} ${lib-ns3-monolib}
          ${LIB_AS_NEEDED_POST}
        )
      else()
        target_link_libraries(
          ${test${BLIB_LIBNAME}} ${LIB_AS_NEEDED_PRE} ${lib${BLIB_LIBNAME}}
          "${BLIB_LIBRARIES_TO_LINK}" ${LIB_AS_NEEDED_POST}
        )
      endif()
      set_target_properties(
        ${test${BLIB_LIBNAME}}
        PROPERTIES OUTPUT_NAME
                   ns${NS3_VER}-${BLIB_LIBNAME}-test${build_profile_suffix}
      )

      target_compile_definitions(
        ${test${BLIB_LIBNAME}} PRIVATE NS_TEST_SOURCEDIR="${FOLDER}/test"
      )
      if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${IGNORE_PCH}))
        target_precompile_headers(${test${BLIB_LIBNAME}} REUSE_FROM stdlib_pch)
      endif()
    endif()
  endif()

  # Build lib examples if requested
  foreach(example_folder example;examples)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_folder})
      if(${ENABLE_EXAMPLES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_folder}/CMakeLists.txt)
          add_subdirectory(${example_folder})
        endif()
      endif()
      scan_python_examples(${CMAKE_CURRENT_SOURCE_DIR}/${example_folder})
    endif()
  endforeach()

  # Get architecture pair for python bindings
  if((${CMAKE_SIZEOF_VOID_P} EQUAL 8) AND (NOT APPLE))
    set(arch gcc_LP64)
    set(arch_flags -m64)
  else()
    set(arch gcc_ILP32)
    set(arch_flags)
  endif()

  # Add target to scan python bindings
  if(${ENABLE_SCAN_PYTHON_BINDINGS}
     AND (EXISTS ${CMAKE_HEADER_OUTPUT_DIRECTORY}/${BLIB_LIBNAME}-module.h)
     AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bindings")
  )
    set(bindings_output_folder ${PROJECT_SOURCE_DIR}/${FOLDER}/bindings)
    file(MAKE_DIRECTORY ${bindings_output_folder})
    set(module_api_ILP32 ${bindings_output_folder}/modulegen__gcc_ILP32.py)
    set(module_api_LP64 ${bindings_output_folder}/modulegen__gcc_LP64.py)

    set(modulescan_modular_command
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/bindings/python/ns3modulescan-modular.py
    )

    set(header_map "")
    # We need a python map that takes header.h to module e.g. "ptr.h": "core"
    foreach(header ${BLIB_HEADER_FILES})
      # header is a relative path to the current working directory
      get_filename_component(
        header_name ${CMAKE_CURRENT_SOURCE_DIR}/${header} NAME
      )
      string(APPEND header_map "\"${header_name}\":\"${BLIB_LIBNAME}\",")
    endforeach()

    set(ns3-headers-to-module-map "${ns3-headers-to-module-map}${header_map}"
        CACHE INTERNAL "Map connecting headers to their modules"
    )

    # API scan needs the include directories to find a few headers (e.g. mpi.h)
    get_target_includes(${lib${BLIB_LIBNAME}} modulegen_include_dirs)

    set(module_to_generate_api ${module_api_ILP32})
    set(LP64toILP32)
    if("${arch}" STREQUAL "gcc_LP64")
      set(module_to_generate_api ${module_api_LP64})
      set(LP64toILP32
          ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/build-support/pybindings-LP64-to-ILP32.py
          ${module_api_LP64} ${module_api_ILP32}
      )
    endif()

    add_custom_target(
      ${lib${BLIB_LIBNAME}}-apiscan
      COMMAND
        ${modulescan_modular_command} ${CMAKE_OUTPUT_DIRECTORY} ${BLIB_LIBNAME}
        ${PROJECT_BINARY_DIR}/header_map.json ${module_to_generate_api}
        \"${arch_flags} ${modulegen_include_dirs}\" 2>
        ${bindings_output_folder}/apiscan.log
      COMMAND ${LP64toILP32}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS ${lib${BLIB_LIBNAME}}
    )
    add_dependencies(apiscan-all ${lib${BLIB_LIBNAME}}-apiscan)
  endif()

  # Build pybindings if requested and if bindings subfolder exists in
  # NS3/src/BLIB_LIBNAME
  if(${ENABLE_PYTHON_BINDINGS} AND (EXISTS
                                    "${CMAKE_CURRENT_SOURCE_DIR}/bindings")
  )
    set(bindings_output_folder ${CMAKE_OUTPUT_DIRECTORY}/${FOLDER}/bindings)
    file(MAKE_DIRECTORY ${bindings_output_folder})
    set(module_src ${bindings_output_folder}/ns3module.cc)
    set(module_hdr ${bindings_output_folder}/ns3module.h)

    string(REPLACE "-" "_" BLIB_LIBNAME_sub ${BLIB_LIBNAME}) # '-' causes
                                                             # problems (e.g.
    # csma-layout), replace with '_' (e.g. csma_layout)

    # Set prefix of binding to _ if a ${BLIB_LIBNAME}.py exists, and copy the
    # ${BLIB_LIBNAME}.py to the output folder
    set(prefix)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bindings/${BLIB_LIBNAME}.py)
      set(prefix _)
      file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/${BLIB_LIBNAME}.py
           DESTINATION ${CMAKE_OUTPUT_DIRECTORY}/bindings/python/ns
      )
    endif()

    # Run modulegen-modular to generate the bindings sources
    if((NOT EXISTS ${module_hdr}) OR (NOT EXISTS ${module_src})) # OR TRUE) # to
                                                                 # force
                                                                 # reprocessing
      string(REPLACE ";" "," ENABLED_FEATURES
                     "${ns3-libs};${BLIB_MODULE_ENABLED_FEATURES}"
      )
      set(modulegen_modular_command
          GCC_RTTI_ABI_COMPLETE=True NS3_ENABLED_FEATURES="${ENABLED_FEATURES}"
          ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/bindings/python/ns3modulegen-modular.py
      )
      execute_process(
        COMMAND
          ${CMAKE_COMMAND} -E env
          PYTHONPATH=${CMAKE_OUTPUT_DIRECTORY}:$ENV{PYTHONPATH}
          ${modulegen_modular_command} ${CMAKE_CURRENT_SOURCE_DIR} ${arch}
          ${prefix}${BLIB_LIBNAME_sub} ${module_src}
        TIMEOUT 60
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_FILE ${module_hdr}
        ERROR_FILE ${bindings_output_folder}/ns3modulegen.log
        RESULT_VARIABLE error_code
      )
      if(${error_code} OR NOT (EXISTS ${module_hdr}))
        # Delete broken bindings to make sure we will his this error again if
        # nothing changed
        if(EXISTS ${module_src})
          file(REMOVE ${module_src})
        endif()
        if(EXISTS ${module_hdr})
          file(REMOVE ${module_hdr})
        endif()
        message(
          FATAL_ERROR
            "Something went wrong during processing of the python bindings of module ${BLIB_LIBNAME}."
            " Make sure you have the latest version of Pybindgen."
        )
      endif()
    endif()

    # Add core module helper sources
    set(python_module_files ${module_hdr} ${module_src})
    file(GLOB custom_python_module_files
         ${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cc
         ${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.h
    )
    list(APPEND python_module_files ${custom_python_module_files})
    set(bindings-name lib${BLIB_LIBNAME}-bindings)
    add_library(${bindings-name} SHARED "${python_module_files}")
    target_include_directories(
      ${bindings-name} PUBLIC ${Python3_INCLUDE_DIRS} ${bindings_output_folder}
    )

    # If there is any, remove the "lib" prefix of libraries (search for
    # "set(lib${BLIB_LIBNAME}")
    list(LENGTH ns_libraries_to_link num_libraries)
    if(num_libraries GREATER "0")
      string(REPLACE ";" "-bindings;" bindings_to_link
                     "${ns_libraries_to_link};"
      ) # add -bindings suffix to all lib${name}
    endif()
    target_link_libraries(
      ${bindings-name}
      PUBLIC ${LIB_AS_NEEDED_PRE} ${lib${BLIB_LIBNAME}} "${bindings_to_link}"
             "${BLIB_LIBRARIES_TO_LINK}" ${LIB_AS_NEEDED_POST}
      PRIVATE ${Python3_LIBRARIES}
    )
    target_include_directories(
      ${bindings-name} PRIVATE ${PROJECT_SOURCE_DIR}/src/core/bindings
    )

    set(suffix)
    if(APPLE)
      # Python doesn't like Apple's .dylib and will refuse to load bindings
      # unless its an .so
      set(suffix SUFFIX .so)
    endif()

    # Set binding library name and output folder
    set_target_properties(
      ${bindings-name}
      PROPERTIES OUTPUT_NAME ${prefix}${BLIB_LIBNAME_sub}
                 PREFIX ""
                 ${suffix} LIBRARY_OUTPUT_DIRECTORY
                 ${CMAKE_OUTPUT_DIRECTORY}/bindings/python/ns
    )

    set(ns3-python-bindings-modules
        "${bindings-name};${ns3-python-bindings-modules}"
        CACHE INTERNAL "list of modules python bindings"
    )

    # Make sure all bindings are built before building the visualizer module
    # that makes use of them
    if(${ENABLE_VISUALIZER} AND (visualizer IN_LIST libs_to_build))
      if(NOT (${BLIB_LIBNAME} STREQUAL visualizer))
        add_dependencies(${libvisualizer} ${bindings-name})
      endif()
    endif()
  endif()

  # Handle package export
  install(
    TARGETS ${lib${BLIB_LIBNAME}}
    EXPORT ns3ExportTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ns3"
  )
  if(${NS3_VERBOSE})
    message(STATUS "Processed ${FOLDER}")
  endif()
endfunction()

# cmake-format: off
#
# This macro processes a ns-3 module example
#
# Arguments:
# NAME = example name (e.g. command-line-example)
# LIBNAME = parent library (e.g. core)
# SOURCE_FILES = "cmake;list;of;.cc;files;"
# HEADER_FILES = "cmake;list;of;public;.h;files;"
# LIBRARIES_TO_LINK = "cmake;list;of;${library_names};"
#
function(build_lib_example)
  # Argument parsing
  set(options IGNORE_PCH)
  set(oneValueArgs NAME)
  set(multiValueArgs SOURCE_FILES HEADER_FILES LIBRARIES_TO_LINK)
  cmake_parse_arguments("BLIB_EXAMPLE" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Get path src/module or contrib/module
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")

  # cmake-format: on
  check_for_missing_libraries(
    missing_dependencies "${BLIB_EXAMPLE_LIBRARIES_TO_LINK}"
  )
  if(NOT missing_dependencies)
    # Create shared library with sources and headers
    add_executable(
      "${BLIB_EXAMPLE_NAME}" ${BLIB_EXAMPLE_SOURCE_FILES}
                             ${BLIB_EXAMPLE_HEADER_FILES}
    )

    if(${NS3_STATIC})
      target_link_libraries(
        ${BLIB_EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE_STATIC} ${lib-ns3-static}
      )
    elseif(${NS3_MONOLIB})
      target_link_libraries(
        ${BLIB_EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE} ${lib-ns3-monolib}
        ${LIB_AS_NEEDED_POST}
      )
    else()
      target_link_libraries(
        ${BLIB_EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE} ${lib${BLIB_EXAMPLE_LIBNAME}}
        ${BLIB_EXAMPLE_LIBRARIES_TO_LINK} ${optional_visualizer_lib}
        ${LIB_AS_NEEDED_POST}
      )
    endif()

    if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${BLIB_EXAMPLE_IGNORE_PCH}))
      target_precompile_headers(${BLIB_EXAMPLE_NAME} REUSE_FROM stdlib_pch_exec)
    endif()

    set_runtime_outputdirectory(
      ${BLIB_EXAMPLE_NAME}
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FOLDER}/ ""
    )
  endif()
endfunction()

# This macro processes a ns-3 module header file (module_name-module.h)
#
# Arguments: name = module name (e.g. core, wifi) HEADER_FILES =
# "cmake;list;of;public;.h;files;"
function(write_module_header name header_files)
  string(TOUPPER "${name}" uppercase_name)
  string(REPLACE "-" "_" final_name "${uppercase_name}")
  # Common module_header
  list(APPEND contents "#ifdef NS3_MODULE_COMPILATION ")
  list(
    APPEND
    contents
    "
    error \"Do not include ns3 module aggregator headers from other modules; these are meant only for end user scripts.\" "
  )
  list(APPEND contents "
#endif "
  )
  list(APPEND contents "
#ifndef NS3_MODULE_"
  )
  list(APPEND contents ${final_name})
  list(APPEND contents "
    // Module headers: "
  )

  # Write each header listed to the contents variable
  foreach(header ${header_files})
    get_filename_component(head ${header} NAME)
    list(APPEND contents "
    #include <ns3/${head}>"
    )
    # include \"ns3/${head}\"")
  endforeach()

  # Common module footer
  list(APPEND contents "
#endif "
  )
  if(EXISTS ${CMAKE_HEADER_OUTPUT_DIRECTORY}/${name}-module.h)
    file(READ ${CMAKE_HEADER_OUTPUT_DIRECTORY}/${name}-module.h oldcontents)
    string(REPLACE ";" "" contents "${contents}")
    # If the header-module.h already exists and is the exact same, do not
    # overwrite it to preserve timestamps
    if("${contents}" STREQUAL "${oldcontents}")
      return()
    endif()
  endif()
  file(WRITE ${CMAKE_HEADER_OUTPUT_DIRECTORY}/${name}-module.h ${contents})
endfunction()
