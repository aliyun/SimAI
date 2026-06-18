set(NS3_WITH_OPENFLOW
    ""
    CACHE PATH
          "Build with Openflow support"
)
set(NS3_OPENFLOW
    "OFF"
    CACHE INTERNAL
          "ON if Openflow is found in NS3_WITH_OPENFLOW"
)

find_external_library(
  DEPENDENCY_NAME openflow
  HEADER_NAME openflow.h
  LIBRARY_NAME openflow
  SEARCH_PATHS ${NS3_WITH_OPENFLOW}
)

if((NOT
    openflow_FOUND)
   AND (NOT
        ${openflow_FOUND})
)
  message(
    ${HIGHLIGHTED_STATUS}
    "Skipping src/openflow"
  )
  return()
endif()

check_include_file_cxx(
  boost/static_assert.hpp
  BOOST_STATIC_ASSERT
)
if(NOT
   BOOST_STATIC_ASSERT
)
  message(STATUS "Openflow requires Boost static_assert.hpp")
  return()
endif()

include_directories(${openflow_INCLUDE_DIRS})
add_definitions(
  -DNS3_OPENFLOW
  -DENABLE_OPENFLOW
)
set(NS3_OPENFLOW
    "ON"
    CACHE INTERNAL
          "ON if Openflow is found in NS3_WITH_OPENFLOW"
)

if(${GCC})
  add_compile_options(-Wno-stringop-truncation) # This option does not exist in CLANG
elseif(${CLANG})
  add_compile_options(-Wno-reserved-user-defined-literal)
endif()

build_lib(
  LIBNAME openflow
  SOURCE_FILES
    helper/openflow-switch-helper.cc
    model/openflow-interface.cc
    model/openflow-switch-net-device.cc
  HEADER_FILES
    helper/openflow-switch-helper.h
    model/openflow-interface.h
    model/openflow-switch-net-device.h
  LIBRARIES_TO_LINK ${libinternet}
                    ${openflow_LIBRARIES}
  TEST_SOURCES test/openflow-switch-test-suite.cc
)
