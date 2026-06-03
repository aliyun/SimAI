build_lib_example(
  NAME three-gpp-http-example
  SOURCE_FILES three-gpp-http-example.cc
  LIBRARIES_TO_LINK
    ${libapplications}
    ${libpoint-to-point}
    ${libinternet}
    ${libnetwork}
)
