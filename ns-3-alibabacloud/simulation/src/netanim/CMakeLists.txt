build_lib(
  LIBNAME netanim
  SOURCE_FILES model/animation-interface.cc
  HEADER_FILES model/animation-interface.h
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libmobility}
    ${libwimax}
    ${libwifi}
    ${libcsma}
    ${liblte}
    ${libuan}
    ${libenergy}
    ${liblr-wpan}
    ${libwave}
    ${libpoint-to-point-layout}
  TEST_SOURCES test/netanim-test.cc
)
