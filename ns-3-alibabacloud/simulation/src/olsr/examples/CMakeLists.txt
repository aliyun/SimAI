build_lib_example(
  NAME olsr-hna
  SOURCE_FILES olsr-hna.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libwifi}
    ${libcsma}
    ${libolsr}
)

build_lib_example(
  NAME simple-point-to-point-olsr
  SOURCE_FILES simple-point-to-point-olsr.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libolsr}
    ${libapplications}
    ${libwifi}
)
