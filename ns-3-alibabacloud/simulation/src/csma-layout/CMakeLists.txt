build_lib(
  LIBNAME csma-layout
  SOURCE_FILES model/csma-star-helper.cc
  HEADER_FILES model/csma-star-helper.h
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libinternet}
    ${libcsma}
    ${libpoint-to-point}
)
