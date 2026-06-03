set(base_examples
    dumbbell-animation
    grid-animation
    star-animation
    colors-link-description
    resources-counters
)

foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
    LIBRARIES_TO_LINK
      ${libnetanim}
      ${libapplications}
      ${libpoint-to-point-layout}
  )
endforeach()

build_lib_example(
  NAME wireless-animation
  SOURCE_FILES wireless-animation.cc
  LIBRARIES_TO_LINK
    ${libnetanim}
    ${libapplications}
    ${libpoint-to-point}
    ${libcsma}
    ${libwifi}
    ${libmobility}
    ${libnetwork}
)

build_lib_example(
  NAME uan-animation
  SOURCE_FILES uan-animation.cc
  LIBRARIES_TO_LINK
    ${libnetanim}
    ${libinternet}
    ${libmobility}
    ${libapplications}
    ${libuan}
)
