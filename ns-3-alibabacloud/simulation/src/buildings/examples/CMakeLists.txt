build_lib_example(
  NAME buildings-pathloss-profiler
  SOURCE_FILES buildings-pathloss-profiler.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libconfig-store}
    ${libmobility}
    ${libbuildings}
)

build_lib_example(
  NAME outdoor-group-mobility-example
  SOURCE_FILES outdoor-group-mobility-example.cc
  LIBRARIES_TO_LINK ${libbuildings}
)

build_lib_example(
  NAME outdoor-random-walk-example
  SOURCE_FILES outdoor-random-walk-example.cc
  LIBRARIES_TO_LINK ${libbuildings}
)
