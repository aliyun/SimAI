build_lib_example(
  NAME li-ion-energy-source
  SOURCE_FILES li-ion-energy-source.cc
  LIBRARIES_TO_LINK ${libcore}
                    ${libenergy}
)

build_lib_example(
  NAME rv-battery-model-test
  SOURCE_FILES rv-battery-model-test.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libenergy}
    ${libwifi}
)

build_lib_example(
  NAME basic-energy-model-test
  SOURCE_FILES basic-energy-model-test.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libenergy}
    ${libwifi}
)
