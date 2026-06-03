if(${ENABLE_TAP})
  build_lib_example(
    NAME tap-csma
    SOURCE_FILES tap-csma.cc
    LIBRARIES_TO_LINK
      ${libcsma}
      ${libtap-bridge}
      ${libinternet}
      ${libwifi}
  )

  build_lib_example(
    NAME tap-csma-virtual-machine
    SOURCE_FILES tap-csma-virtual-machine.cc
    LIBRARIES_TO_LINK
      ${libcsma}
      ${libtap-bridge}
      ${libinternet}
  )

  build_lib_example(
    NAME tap-wifi-virtual-machine
    SOURCE_FILES tap-wifi-virtual-machine.cc
    LIBRARIES_TO_LINK
      ${libcsma}
      ${libtap-bridge}
      ${libinternet}
      ${libwifi}
      ${libmobility}
  )

  build_lib_example(
    NAME tap-wifi-dumbbell
    SOURCE_FILES tap-wifi-dumbbell.cc
    LIBRARIES_TO_LINK
      ${libwifi}
      ${libcsma}
      ${libpoint-to-point}
      ${libtap-bridge}
      ${libinternet}
      ${libapplications}
  )

endif()
