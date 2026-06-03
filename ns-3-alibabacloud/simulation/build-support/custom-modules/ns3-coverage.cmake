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

if(${NS3_COVERAGE})
  mark_as_advanced(GCOV)
  find_program(GCOV gcov)
  if(NOT ("${GCOV}" STREQUAL "GCOV-NOTFOUND"))
    add_definitions(--coverage)
    link_libraries(-lgcov)
  endif()

  mark_as_advanced(LCOV)
  find_program(LCOV lcov)
  if("${LCOV}" STREQUAL "LCOV-NOTFOUND")
    message(FATAL_ERROR "LCOV is required but it is not installed.")
  endif()

  if(${NS3_COVERAGE_ZERO_COUNTERS})
    set(zero_counters "--lcov-zerocounters")
  endif()
  # The following target will run test.py --no-build to generate the code
  # coverage files .gcno and .gcda output will be in ${CMAKE_BINARY_DIR} a.k.a.
  # cmake-cache or cmake-build-${build_suffix}

  # Create output directory for coverage info and html
  file(MAKE_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY}/coverage)

  # Extract code coverage results and build html report
  add_custom_target(
    coverage_gcc
    COMMAND lcov -o ns3.info -c --directory ${CMAKE_BINARY_DIR} ${zero_counters}
    COMMAND genhtml ns3.info
    WORKING_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY}/coverage
    DEPENDS run_test_py
  )
endif()
