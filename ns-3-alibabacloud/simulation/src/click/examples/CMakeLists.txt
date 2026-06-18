build_lib_example(
  NAME nsclick-simple-lan
  SOURCE_FILES nsclick-simple-lan.cc
  LIBRARIES_TO_LINK
    ${libclick}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_lib_example(
  NAME nsclick-raw-wlan
  SOURCE_FILES nsclick-raw-wlan.cc
  LIBRARIES_TO_LINK
    ${libclick}
    ${libwifi}
    ${libinternet}
    ${libapplications}
)

build_lib_example(
  NAME nsclick-udp-client-server-csma
  SOURCE_FILES nsclick-udp-client-server-csma.cc
  LIBRARIES_TO_LINK
    ${libclick}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_lib_example(
  NAME nsclick-udp-client-server-wifi
  SOURCE_FILES nsclick-udp-client-server-wifi.cc
  LIBRARIES_TO_LINK
    ${libclick}
    ${libwifi}
    ${libinternet}
    ${libapplications}
)

build_lib_example(
  NAME nsclick-routing
  SOURCE_FILES nsclick-routing.cc
  LIBRARIES_TO_LINK
    ${libclick}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)

build_lib_example(
  NAME nsclick-defines
  SOURCE_FILES nsclick-defines.cc
  LIBRARIES_TO_LINK ${libclick}
)
