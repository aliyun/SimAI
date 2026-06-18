build_example(
  NAME hello-simulator
  SOURCE_FILES hello-simulator.cc
  LIBRARIES_TO_LINK ${libcore}
)

build_example(
  NAME first
  SOURCE_FILES first.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME second
  SOURCE_FILES second.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libpoint-to-point}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME third
  SOURCE_FILES third.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libpoint-to-point}
    ${libcsma}
    ${libwifi}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME fourth
  SOURCE_FILES fourth.cc
  LIBRARIES_TO_LINK ${libcore}
)

build_example(
  NAME fifth
  SOURCE_FILES fifth.cc
               tutorial-app.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME sixth
  SOURCE_FILES sixth.cc
               tutorial-app.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME seventh
  SOURCE_FILES seventh.cc
               tutorial-app.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libstats}
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)
