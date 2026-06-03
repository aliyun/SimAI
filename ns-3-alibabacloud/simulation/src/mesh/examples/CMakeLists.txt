build_lib_example(
  NAME mesh
  SOURCE_FILES mesh.cc
  LIBRARIES_TO_LINK
    ${libinternet}
    ${libmobility}
    ${libwifi}
    ${libmesh}
    ${libapplications}
)
