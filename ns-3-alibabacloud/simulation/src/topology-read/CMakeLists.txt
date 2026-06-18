build_lib(
  LIBNAME topology-read
  SOURCE_FILES
    helper/topology-reader-helper.cc
    model/inet-topology-reader.cc
    model/orbis-topology-reader.cc
    model/rocketfuel-topology-reader.cc
    model/topology-reader.cc
  HEADER_FILES
    helper/topology-reader-helper.h
    model/inet-topology-reader.h
    model/orbis-topology-reader.h
    model/rocketfuel-topology-reader.h
    model/topology-reader.h
  LIBRARIES_TO_LINK ${libnetwork}
                    ${PCRE_LIBRARIES}
  TEST_SOURCES test/rocketfuel-topology-reader-test-suite.cc
)
