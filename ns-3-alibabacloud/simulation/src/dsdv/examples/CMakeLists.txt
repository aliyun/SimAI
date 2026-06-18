build_lib_example(
  NAME dsdv-manet
  SOURCE_FILES dsdv-manet.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libinternet}
    ${libdsdv}
    ${libapplications}
)
