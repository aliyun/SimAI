build_lib_example(
  NAME simple-mtp
  SOURCE_FILES simple-mtp.cc
  LIBRARIES_TO_LINK
    ${libmtp}
    ${libpoint-to-point}
    ${libinternet}
    ${libnix-vector-routing}
    ${libapplications}
)
