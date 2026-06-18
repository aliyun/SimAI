build_example(
  NAME dynamic-global-routing
  SOURCE_FILES dynamic-global-routing.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME static-routing-slash32
  SOURCE_FILES static-routing-slash32.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME global-routing-slash32
  SOURCE_FILES global-routing-slash32.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME global-injection-slash32
  SOURCE_FILES global-injection-slash32.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME simple-global-routing
  SOURCE_FILES simple-global-routing.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
)

build_example(
  NAME simple-alternate-routing
  SOURCE_FILES simple-alternate-routing.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME mixed-global-routing
  SOURCE_FILES mixed-global-routing.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libcsma}
    ${libapplications}
)

build_example(
  NAME simple-routing-ping6
  SOURCE_FILES simple-routing-ping6.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libinternet-apps}
)

build_example(
  NAME manet-routing-compare
  SOURCE_FILES manet-routing-compare.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libdsr}
    ${libdsdv}
    ${libaodv}
    ${libolsr}
    ${libinternet}
    ${libapplications}
)

build_example(
  NAME ripng-simple-network
  SOURCE_FILES ripng-simple-network.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libinternet-apps}
)

build_example(
  NAME rip-simple-network
  SOURCE_FILES rip-simple-network.cc
  LIBRARIES_TO_LINK
    ${libcsma}
    ${libinternet}
    ${libinternet-apps}
)

build_example(
  NAME global-routing-multi-switch-plus-router
  SOURCE_FILES global-routing-multi-switch-plus-router.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libapplications}
    ${libinternet}
    ${libbridge}
    ${libcsma}
    ${libpoint-to-point}
    ${libcsma}
    ${libinternet}
)

build_example(
  NAME simple-multicast-flooding
  SOURCE_FILES simple-multicast-flooding.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libapplications}
    ${libinternet}
)
