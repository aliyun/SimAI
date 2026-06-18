if(${ENABLE_MTP})
  build_example(
    NAME dctcp-example-mtp
    SOURCE_FILES dctcp-example-mtp.cc
    LIBRARIES_TO_LINK
      ${libcore}
      ${libnetwork}
      ${libinternet}
      ${libpoint-to-point}
      ${libapplications}
      ${libtraffic-control}
      ${libmtp}
  )

  build_example(
    NAME dynamic-global-routing-mtp
    SOURCE_FILES dynamic-global-routing-mtp.cc
    LIBRARIES_TO_LINK
      ${libpoint-to-point}
      ${libcsma}
      ${libinternet}
      ${libapplications}
      ${libmtp}
  )

  build_example(
    NAME queue-discs-benchmark-mtp
    SOURCE_FILES queue-discs-benchmark-mtp.cc
    LIBRARIES_TO_LINK
      ${libinternet}
      ${libpoint-to-point}
      ${libapplications}
      ${libinternet-apps}
      ${libtraffic-control}
      ${libflow-monitor}
      ${libmtp}
  )

  build_example(
    NAME ripng-simple-network-mtp
    SOURCE_FILES ripng-simple-network-mtp.cc
    LIBRARIES_TO_LINK
      ${libcsma}
      ${libinternet}
      ${libpoint-to-point}
      ${libinternet-apps}
      ${libmtp}
  )

  build_example(
    NAME simple-multicast-flooding-mtp
    SOURCE_FILES simple-multicast-flooding-mtp.cc
    LIBRARIES_TO_LINK
      ${libcore}
      ${libnetwork}
      ${libapplications}
      ${libinternet}
      ${libmtp}
  )

  build_example(
    NAME socket-bound-tcp-static-routing-mtp
    SOURCE_FILES socket-bound-tcp-static-routing-mtp.cc
    LIBRARIES_TO_LINK
      ${libnetwork}
      ${libcsma}
      ${libpoint-to-point}
      ${libinternet}
      ${libapplications}
      ${libmtp}
  )

  build_example(
    NAME tcp-bbr-example-mtp
    SOURCE_FILES tcp-bbr-example-mtp.cc
    LIBRARIES_TO_LINK
      ${libpoint-to-point}
      ${libinternet}
      ${libapplications}
      ${libtraffic-control}
      ${libnetwork}
      ${libinternet-apps}
      ${libflow-monitor}
      ${libmtp}
  )

  build_example(
    NAME tcp-pacing-mtp
    SOURCE_FILES tcp-pacing-mtp.cc
    LIBRARIES_TO_LINK
      ${libpoint-to-point}
      ${libinternet}
      ${libapplications}
      ${libflow-monitor}
      ${libmtp}
  )

  build_example(
    NAME tcp-star-server-mtp
    SOURCE_FILES tcp-star-server-mtp.cc
    LIBRARIES_TO_LINK
      ${libpoint-to-point}
      ${libapplications}
      ${libinternet}
      ${libmtp}
  )

  build_example(
    NAME tcp-validation-mtp
    SOURCE_FILES tcp-validation-mtp.cc
    LIBRARIES_TO_LINK
      ${libpoint-to-point}
      ${libinternet}
      ${libapplications}
      ${libtraffic-control}
      ${libnetwork}
      ${libinternet-apps}
      ${libmtp}
  )
endif()
