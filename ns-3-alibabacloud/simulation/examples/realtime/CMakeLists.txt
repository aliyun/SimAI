if(${ENABLE_REALTIME})
  build_example(
    NAME realtime-udp-echo
    SOURCE_FILES realtime-udp-echo.cc
    LIBRARIES_TO_LINK
      ${libcsma}
      ${libinternet}
      ${libapplications}
  )
endif()
