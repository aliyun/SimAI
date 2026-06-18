build_lib_example(
  NAME csma-star
  SOURCE_FILES csma-star.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libcsma-layout}
    ${libinternet}
    ${libapplications}
)
