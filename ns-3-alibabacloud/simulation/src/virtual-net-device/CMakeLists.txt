build_lib(
  LIBNAME virtual-net-device
  SOURCE_FILES model/virtual-net-device.cc
  HEADER_FILES model/virtual-net-device.h
  LIBRARIES_TO_LINK ${libnetwork}
)
