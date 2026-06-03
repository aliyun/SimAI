set(base_examples
    bit-serializer
    main-packet-header
    main-packet-tag
    packet-socket-apps
    lollipop-comparisions
)

foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK ${libnetwork}
                      ${libcore}
  )
endforeach()
