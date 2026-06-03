set(base_examples
    command-line-example
    fatal-example
    hash-example
    length-example
    main-callback
    main-ptr
    sample-log-time-format
    sample-random-variable
    sample-random-variable-stream
    sample-show-progress
    sample-simulator
    system-path-examples
    test-string-value-formatting
)

foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK ${libcore}
  )
endforeach()

build_lib_example(
  NAME main-random-variable-stream
  SOURCE_FILES main-random-variable-stream.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libconfig-store}
    ${libstats}
)

if(${ENABLE_BUILD_VERSION})
  build_lib_example(
    NAME build-version-example
    SOURCE_FILES build-version-example.cc
    LIBRARIES_TO_LINK ${libcore}
  )
endif()

if(${ENABLE_REALTIME})
  build_lib_example(
    NAME main-test-sync
    SOURCE_FILES main-test-sync.cc
    LIBRARIES_TO_LINK ${libnetwork}
                      ${CMAKE_THREAD_LIBS_INIT}
  )
endif()

build_lib_example(
  NAME empirical-random-variable-example
  SOURCE_FILES empirical-random-variable-example.cc
  LIBRARIES_TO_LINK ${libcore}
                    ${libflow-monitor}
)
