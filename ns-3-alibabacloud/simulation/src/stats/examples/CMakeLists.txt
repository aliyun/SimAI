build_lib_example(
  NAME time-probe-example
  SOURCE_FILES time-probe-example.cc
  LIBRARIES_TO_LINK ${libstats}
)

set(base_examples
    gnuplot-example
    double-probe-example
    gnuplot-aggregator-example
    gnuplot-helper-example
    file-aggregator-example
    file-helper-example
)

foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK ${libnetwork}
                      ${libstats}
  )
endforeach()
