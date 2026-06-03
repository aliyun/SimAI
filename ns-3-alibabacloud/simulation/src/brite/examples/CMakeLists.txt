build_lib_example(
  NAME brite-generic-example
  SOURCE_FILES brite-generic-example.cc
  LIBRARIES_TO_LINK
    ${libbrite}
    ${libinternet}
    ${libpoint-to-point}
    ${libnix-vector-routing}
    ${libapplications}
)

if(${ENABLE_MPI})
  build_lib_example(
    NAME brite-MPI-example
    SOURCE_FILES brite-MPI-example.cc
    LIBRARIES_TO_LINK
      ${libbrite}
      ${libinternet}
      ${libpoint-to-point}
      ${libnix-vector-routing}
      ${libapplications}
      ${libmpi}
      ${MPI_CXX_LIBRARIES}
  )
endif()
