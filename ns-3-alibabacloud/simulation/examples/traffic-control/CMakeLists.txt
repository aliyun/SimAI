build_example(
  NAME traffic-control
  SOURCE_FILES traffic-control.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libtraffic-control}
    ${libflow-monitor}
)

build_example(
  NAME queue-discs-benchmark
  SOURCE_FILES queue-discs-benchmark.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libinternet-apps}
    ${libtraffic-control}
    ${libflow-monitor}
)

build_example(
  NAME red-vs-fengadaptive
  SOURCE_FILES red-vs-fengadaptive.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libpoint-to-point}
    ${libpoint-to-point-layout}
    ${libapplications}
    ${libtraffic-control}
)

build_example(
  NAME red-vs-nlred
  SOURCE_FILES red-vs-nlred.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libpoint-to-point}
    ${libpoint-to-point-layout}
    ${libapplications}
    ${libtraffic-control}
)

build_example(
  NAME tbf-example
  SOURCE_FILES tbf-example.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libtraffic-control}
)

build_example(
  NAME cobalt-vs-codel
  SOURCE_FILES cobalt-vs-codel.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libtraffic-control}
)
