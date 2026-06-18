build_lib_example(
  NAME main-simple
  SOURCE_FILES main-simple.cc
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${libinternet}
    ${libapplications}
)
