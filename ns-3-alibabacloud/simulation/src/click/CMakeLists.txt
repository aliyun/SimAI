set(NS3_WITH_CLICK
    ""
    CACHE PATH
          "Build with click support"
)
set(NS3_CLICK
    "OFF"
    CACHE INTERNAL
          "ON if Click is found in NS3_WITH_CLICK"
)

find_external_library(
  DEPENDENCY_NAME click
  HEADER_NAME simclick.h
  LIBRARY_NAMES nsclick
                click
  SEARCH_PATHS ${NS3_WITH_CLICK}
)

if((NOT
    click_FOUND)
   AND (NOT
        ${click_FOUND})
)
  message(
    ${HIGHLIGHTED_STATUS}
    "Skipping src/click"
  )
  return()
endif()

include_directories(${click_INCLUDE_DIRS})
set(NS3_CLICK
    "ON"
    CACHE INTERNAL
          "ON if Click is found in NS3_WITH_CLICK"
)
add_definitions(-DNS3_CLICK)

build_lib(
  LIBNAME click
  SOURCE_FILES
    helper/click-internet-stack-helper.cc
    model/ipv4-click-routing.cc
    model/ipv4-l3-click-protocol.cc
  HEADER_FILES
    helper/click-internet-stack-helper.h
    model/ipv4-click-routing.h
    model/ipv4-l3-click-protocol.h
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${click_LIBRARIES}
  TEST_SOURCES test/ipv4-click-routing-test.cc
)
