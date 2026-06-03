build_example(
  NAME socket-bound-static-routing
  SOURCE_FILES socket-bound-static-routing.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libcsma}
    ${libpoint-to-point}
    ${libinternet}
)

build_example(
  NAME socket-bound-tcp-static-routing
  SOURCE_FILES socket-bound-tcp-static-routing.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libcsma}
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME socket-options-ipv4
  SOURCE_FILES socket-options-ipv4.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libcsma}
    ${libpoint-to-point}
    ${libinternet}
)

build_example(
  NAME socket-options-ipv6
  SOURCE_FILES socket-options-ipv6.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libcsma}
    ${libpoint-to-point}
    ${libinternet}
)
