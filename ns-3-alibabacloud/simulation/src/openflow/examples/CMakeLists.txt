build_lib_example(
  NAME openflow-switch
  SOURCE_FILES openflow-switch.cc
  LIBRARIES_TO_LINK
    ${libopenflow}
    ${libinternet}
    ${libcsma}
    ${libapplications}
)
