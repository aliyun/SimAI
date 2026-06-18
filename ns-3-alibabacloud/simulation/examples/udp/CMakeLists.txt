build_example(
  NAME udp-echo
  SOURCE_FILES udp-echo.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libapplications}
)
