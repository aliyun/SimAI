build_example(
  NAME simple-error-model
  SOURCE_FILES simple-error-model.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)
