build_example(
  NAME matrix-topology
  SOURCE_FILES matrix-topology.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libnetanim}
    ${libmobility}
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)
