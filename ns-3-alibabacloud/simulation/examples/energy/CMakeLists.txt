build_example(
  NAME energy-model-example
  SOURCE_FILES energy-model-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libwifi}
    ${libenergy}
    ${libinternet}
    ${libconfig-store}
)

build_example(
  NAME energy-model-with-harvesting-example
  SOURCE_FILES energy-model-with-harvesting-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libwifi}
    ${libenergy}
    ${libinternet}
    ${libconfig-store}
)
