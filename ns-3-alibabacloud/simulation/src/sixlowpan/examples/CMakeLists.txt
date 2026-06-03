build_lib_example(
  NAME example-sixlowpan
  SOURCE_FILES example-sixlowpan.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libsixlowpan}
    ${libinternet}
    ${libcsma}
    ${libinternet-apps}
)

build_lib_example(
  NAME example-ping-lr-wpan
  SOURCE_FILES example-ping-lr-wpan.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libsixlowpan}
    ${libinternet}
    ${liblr-wpan}
    ${libinternet-apps}
)

build_lib_example(
  NAME example-ping-lr-wpan-beacon
  SOURCE_FILES example-ping-lr-wpan-beacon.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libsixlowpan}
    ${libinternet}
    ${liblr-wpan}
    ${libinternet-apps}
)

build_lib_example(
  NAME example-ping-lr-wpan-mesh-under
  SOURCE_FILES example-ping-lr-wpan-mesh-under.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libsixlowpan}
    ${libinternet}
    ${liblr-wpan}
    ${libinternet-apps}
    ${libcsma}
)
