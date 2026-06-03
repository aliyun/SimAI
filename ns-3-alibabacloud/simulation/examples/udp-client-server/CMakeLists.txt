build_example(
  NAME udp-client-server
  SOURCE_FILES udp-client-server.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME udp-trace-client-server
  SOURCE_FILES udp-trace-client-server.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libapplications}
)
