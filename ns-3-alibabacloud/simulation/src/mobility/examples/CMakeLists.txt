set(base_examples
    bonnmotion-ns2-example
    main-random-topology
    main-random-walk
    ns2-mobility-trace
)
foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK ${libcore}
                      ${libmobility}
  )
endforeach()

build_lib_example(
  NAME main-grid-topology
  SOURCE_FILES main-grid-topology.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libnetwork}
)

build_lib_example(
  NAME mobility-trace-example
  SOURCE_FILES mobility-trace-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libnetwork}
)

build_lib_example(
  NAME reference-point-group-mobility-example
  SOURCE_FILES reference-point-group-mobility-example.cc
  LIBRARIES_TO_LINK ${libnetwork}
                    ${libmobility}
)
