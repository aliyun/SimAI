build_lib_example(
  NAME wimax-ipv4
  SOURCE_FILES wimax-ipv4.cc
  LIBRARIES_TO_LINK
    ${libwimax}
    ${libinternet}
    ${libmobility}
    ${libcsma}
    ${libapplications}
)

build_lib_example(
  NAME wimax-multicast
  SOURCE_FILES wimax-multicast.cc
  LIBRARIES_TO_LINK
    ${libwimax}
    ${libinternet}
    ${libcsma}
    ${libapplications}
)

build_lib_example(
  NAME wimax-simple
  SOURCE_FILES wimax-simple.cc
  LIBRARIES_TO_LINK
    ${libwimax}
    ${libinternet}
    ${libmobility}
    ${libapplications}
)
