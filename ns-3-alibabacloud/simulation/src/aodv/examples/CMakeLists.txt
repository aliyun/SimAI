build_lib_example(
  NAME aodv
  SOURCE_FILES aodv.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libinternet}
    ${libaodv}
    ${libinternet-apps}
)
