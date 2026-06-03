set(base_examples
    uan-cw-example
    uan-rc-example
    uan-ipv4-example
    uan-ipv6-example
    uan-raw-example
)
foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK
      ${libinternet}
      ${libmobility}
      ${libstats}
      ${libapplications}
      ${libuan}
  )
endforeach()

build_lib_example(
  NAME uan-6lowpan-example
  SOURCE_FILES uan-6lowpan-example.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libmobility}
    ${libstats}
    ${libapplications}
    ${libuan}
    ${libsixlowpan}
)
