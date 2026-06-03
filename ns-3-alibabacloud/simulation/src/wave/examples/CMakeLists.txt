build_lib_example(
  NAME wave-simple-80211p
  SOURCE_FILES wave-simple-80211p.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libapplications}
    ${libmobility}
    ${libnetwork}
    ${libwifi}
    ${libwave}
)

build_lib_example(
  NAME wave-simple-device
  SOURCE_FILES wave-simple-device.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libapplications}
    ${libmobility}
    ${libnetwork}
    ${libwifi}
    ${libwave}
)

build_lib_example(
  NAME vanet-routing-compare
  SOURCE_FILES vanet-routing-compare.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libaodv}
    ${libapplications}
    ${libdsr}
    ${libdsdv}
    ${libflow-monitor}
    ${libmobility}
    ${libnetwork}
    ${libolsr}
    ${libpropagation}
    ${libwifi}
    ${libwave}
)
