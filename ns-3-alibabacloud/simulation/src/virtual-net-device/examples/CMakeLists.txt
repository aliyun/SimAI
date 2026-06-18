build_lib_example(
  NAME virtual-net-device
  SOURCE_FILES virtual-net-device.cc
  LIBRARIES_TO_LINK
    ${libvirtual-net-device}
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
)
