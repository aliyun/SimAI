build_example(
  NAME wifi-example-sim
  SOURCE_FILES wifi-example-sim.cc
               wifi-example-apps.cc
  HEADER_FILES wifi-example-apps.h
  LIBRARIES_TO_LINK
    ${libstats}
    ${libinternet}
    ${libmobility}
    ${libwifi}
)
