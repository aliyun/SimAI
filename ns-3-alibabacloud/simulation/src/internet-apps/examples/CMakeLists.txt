build_lib_example(
  NAME dhcp-example
  SOURCE_FILES dhcp-example.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libinternet-apps}
    ${libcsma}
    ${libpoint-to-point}
    ${libapplications}
)

build_lib_example(
  NAME traceroute-example
  SOURCE_FILES traceroute-example.cc
  LIBRARIES_TO_LINK
    ${libaodv}
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libwifi}
    ${libinternet-apps}
)
