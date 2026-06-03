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

function(cache_cmake_flag cmake_flag cache_entry output_string)
  if(${${cmake_flag}})
    set(${output_string} "${${output_string}}${cache_entry} = True\n"
        PARENT_SCOPE
    )
  else()
    set(${output_string} "${${output_string}}${cache_entry} = False\n"
        PARENT_SCOPE
    )
  endif()
endfunction(cache_cmake_flag)

function(write_lock)
  set(lock_contents "#! /usr/bin/env python3\n\n")

  # Contents previously in ns-3-dev/.lock-waf_sys.platform_build
  string(APPEND lock_contents "launch_dir = '${PROJECT_SOURCE_DIR}'\n")
  string(APPEND lock_contents "run_dir = '${PROJECT_SOURCE_DIR}'\n")
  string(APPEND lock_contents "top_dir = '${PROJECT_SOURCE_DIR}'\n")
  string(APPEND lock_contents "out_dir = '${CMAKE_OUTPUT_DIRECTORY}'\n")
  string(APPEND lock_contents "\n\n")

  # Contents previously in ns-3-dev/build/c4che/_cache.py
  string(APPEND lock_contents "NS3_ENABLED_MODULES = [")
  foreach(module_library ${ns3-libs}) # fetch core module libraries
    string(APPEND lock_contents "'")
    remove_lib_prefix("${module_library}" module_name)
    string(APPEND lock_contents "ns3-${module_name}', ")
  endforeach()
  string(APPEND lock_contents "]\n")

  string(APPEND lock_contents "NS3_ENABLED_CONTRIBUTED_MODULES = [")
  foreach(module_library ${ns3-contrib-libs}) # fetch core module libraries
    string(APPEND lock_contents "'")
    remove_lib_prefix("${module_library}" module_name)
    string(APPEND lock_contents "ns3-${module_name}', ")
  endforeach()
  string(APPEND lock_contents "]\n")

  string(REPLACE ":" "', '" PATH_LIST $ENV{PATH})
  string(
    APPEND
    lock_contents
    "NS3_MODULE_PATH = ['${PATH_LIST}', '${CMAKE_OUTPUT_DIRECTORY}', '${CMAKE_LIBRARY_OUTPUT_DIRECTORY}']\n"
  )

  cache_cmake_flag(ENABLE_REALTIME "ENABLE_REAL_TIME" lock_contents)
  cache_cmake_flag(ENABLE_EXAMPLES "ENABLE_EXAMPLES" lock_contents)
  cache_cmake_flag(ENABLE_TESTS "ENABLE_TESTS" lock_contents)
  cache_cmake_flag(NS3_OPENFLOW "ENABLE_OPENFLOW" lock_contents)
  cache_cmake_flag(NS3_CLICK "NSCLICK" lock_contents)
  cache_cmake_flag(NS3_BRITE "ENABLE_BRITE" lock_contents)
  cache_cmake_flag(NS3_ENABLE_SUDO "ENABLE_SUDO" lock_contents)
  cache_cmake_flag(NS3_PYTHON_BINDINGS "ENABLE_PYTHON_BINDINGS" lock_contents)
  cache_cmake_flag(
    NS3_SCAN_PYTHON_BINDINGS "ENABLE_SCAN_PYTHON_BINDINGS" lock_contents
  )

  string(APPEND lock_contents "EXAMPLE_DIRECTORIES = [")
  foreach(example_folder ${ns3-example-folders})
    string(APPEND lock_contents "'${example_folder}', ")
  endforeach()
  string(APPEND lock_contents "]\n")

  string(APPEND lock_contents "APPNAME = 'ns'\n")
  string(APPEND lock_contents "BUILD_PROFILE = '${build_profile}'\n")
  string(APPEND lock_contents "VERSION = '${NS3_VER}' \n")
  string(APPEND lock_contents "PYTHON = ['${Python3_EXECUTABLE}']\n")

  mark_as_advanced(VALGRIND)
  find_program(VALGRIND valgrind)
  if("${VALGRIND}" STREQUAL "VALGRIND-NOTFOUND")
    string(APPEND lock_contents "VALGRIND_FOUND = False \n")
  else()
    string(APPEND lock_contents "VALGRIND_FOUND = True \n")
  endif()

  # Contents previously in ns-3-dev/build/build-status.py
  string(APPEND lock_contents "\n\n")
  string(APPEND lock_contents "ns3_runnable_programs = [")
  foreach(executable ${ns3-execs})
    string(APPEND lock_contents "'${executable}', ")
  endforeach()
  string(APPEND lock_contents "]\n\n")

  string(APPEND lock_contents "ns3_runnable_scripts = [")
  foreach(executable ${ns3-execs-py})
    string(APPEND lock_contents "'${executable}', ")
  endforeach()
  string(APPEND lock_contents "]\n\n")

  if(LINUX)
    set(lock_filename .lock-ns3_linux_build)
  elseif(APPLE)
    set(lock_filename .lock-ns3_darwin_build)
  elseif(WIN32)
    set(lock_filename .lock-ns3_win32_build)
  else()
    message(FATAL_ERROR "Platform not supported")
  endif()
  file(WRITE ${PROJECT_SOURCE_DIR}/${lock_filename} ${lock_contents})
endfunction(write_lock)
