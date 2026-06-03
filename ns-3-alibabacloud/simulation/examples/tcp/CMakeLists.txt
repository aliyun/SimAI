build_example(
  NAME tcp-large-transfer
  SOURCE_FILES tcp-large-transfer.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libapplications}
    ${libinternet}
)

build_example(
  NAME tcp-star-server
  SOURCE_FILES tcp-star-server.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libapplications}
    ${libinternet}
)

build_example(
  NAME star
  SOURCE_FILES star.cc
  LIBRARIES_TO_LINK
    ${libnetanim}
    ${libpoint-to-point}
    ${libpoint-to-point-layout}
    ${libapplications}
    ${libinternet}
)

build_example(
  NAME tcp-bbr-example
  SOURCE_FILES tcp-bbr-example.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libtraffic-control}
    ${libnetwork}
    ${libinternet-apps}
    ${libflow-monitor}
)

build_example(
  NAME tcp-bulk-send
  SOURCE_FILES tcp-bulk-send.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libapplications}
    ${libinternet}
)

build_example(
  NAME tcp-pcap-nanosec-example
  SOURCE_FILES tcp-pcap-nanosec-example.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libapplications}
    ${libinternet}
)

build_example(
  NAME tcp-variants-comparison
  SOURCE_FILES tcp-variants-comparison.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
)

build_example(
  NAME tcp-pacing
  SOURCE_FILES tcp-pacing.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
)

build_example(
  NAME tcp-linux-reno
  SOURCE_FILES tcp-linux-reno.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libtraffic-control}
    ${libnetwork}
)

build_example(
  NAME tcp-validation
  SOURCE_FILES tcp-validation.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libtraffic-control}
    ${libnetwork}
    ${libinternet-apps}
)

build_example(
  NAME dctcp-example
  SOURCE_FILES dctcp-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libtraffic-control}
)
