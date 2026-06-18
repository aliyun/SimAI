if(PCRE_FOUND)
  build_lib_example(
    NAME topology-example-sim
    SOURCE_FILES topology-example-sim.cc
    LIBRARIES_TO_LINK
      ${libtopology-read}
      ${libinternet}
      ${libnix-vector-routing}
      ${libpoint-to-point}
      ${libapplications}
  )
endif()
