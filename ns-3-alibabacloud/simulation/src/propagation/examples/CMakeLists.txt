build_lib_example(
  NAME main-propagation-loss
  SOURCE_FILES main-propagation-loss.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libpropagation}
    ${libconfig-store}
    ${libstats}
    ${libmobility}
    ${libbuildings}
)

build_lib_example(
  NAME jakes-propagation-model-example
  SOURCE_FILES jakes-propagation-model-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libpropagation}
    ${libbuildings}
)
