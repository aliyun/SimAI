build_lib_example(
  NAME red-tests
  SOURCE_FILES red-tests.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
    ${libtraffic-control}
)

build_lib_example(
  NAME red-vs-ared
  SOURCE_FILES red-vs-ared.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libpoint-to-point-layout}
    ${libinternet}
    ${libapplications}
    ${libtraffic-control}
)

build_lib_example(
  NAME adaptive-red-tests
  SOURCE_FILES adaptive-red-tests.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
    ${libtraffic-control}
)

build_lib_example(
  NAME pfifo-vs-red
  SOURCE_FILES pfifo-vs-red.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libpoint-to-point-layout}
    ${libinternet}
    ${libapplications}
    ${libtraffic-control}
)

build_lib_example(
  NAME codel-vs-pfifo-basic-test
  SOURCE_FILES codel-vs-pfifo-basic-test.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libnetwork}
    ${libinternet}
    ${libapplications}
    ${libtraffic-control}
)

build_lib_example(
  NAME codel-vs-pfifo-asymmetric
  SOURCE_FILES codel-vs-pfifo-asymmetric.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libnetwork}
    ${libinternet}
    ${libapplications}
    ${libtraffic-control}
)

build_lib_example(
  NAME pie-example
  SOURCE_FILES pie-example.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
    ${libtraffic-control}
)

build_lib_example(
  NAME fqcodel-l4s-example
  SOURCE_FILES fqcodel-l4s-example.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libinternet-apps}
    ${libapplications}
    ${libflow-monitor}
    ${libtraffic-control}
)
