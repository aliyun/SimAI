build_lib_example(
  NAME wifi-phy-test
  SOURCE_FILES wifi-phy-test.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libnetwork}
    ${libwifi}
)

build_lib_example(
  NAME wifi-test-interference-helper
  SOURCE_FILES wifi-test-interference-helper.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libmobility}
    ${libnetwork}
    ${libwifi}
)

build_lib_example(
  NAME wifi-manager-example
  SOURCE_FILES wifi-manager-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libwifi}
    ${libstats}
    ${libmobility}
    ${libpropagation}
)

build_lib_example(
  NAME wifi-trans-example
  SOURCE_FILES wifi-trans-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libwifi}
    ${libstats}
    ${libmobility}
    ${libpropagation}
)

build_lib_example(
  NAME wifi-phy-configuration
  SOURCE_FILES wifi-phy-configuration.cc
  LIBRARIES_TO_LINK ${libwifi}
                    ${libconfig-store}
)

build_lib_example(
  NAME wifi-bianchi
  SOURCE_FILES wifi-bianchi.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libapplications}
    ${libinternet-apps}
)
