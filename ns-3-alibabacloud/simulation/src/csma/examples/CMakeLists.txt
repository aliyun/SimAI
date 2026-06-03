set(base_examples
    csma-one-subnet
    csma-broadcast
    csma-packet-socket
    csma-multicast
)
foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK
      ${libcsma}
      ${libinternet}
      ${libapplications}
  )
endforeach()

build_lib_example(
  NAME csma-raw-ip-socket
  SOURCE_FILES csma-raw-ip-socket.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libapplications}
    ${libinternet-apps}
)

build_lib_example(
  NAME csma-ping
  SOURCE_FILES csma-ping.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libapplications}
    ${libinternet-apps}
)
