build_lib_example(
  NAME csma-bridge
  SOURCE_FILES csma-bridge.cc
  LIBRARIES_TO_LINK
    ${libbridge}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_lib_example(
  NAME csma-bridge-one-hop
  SOURCE_FILES csma-bridge-one-hop.cc
  LIBRARIES_TO_LINK
    ${libbridge}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)
