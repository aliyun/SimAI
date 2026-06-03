build_lib(
  LIBNAME bridge
  SOURCE_FILES
    helper/bridge-helper.cc
    model/bridge-channel.cc
    model/bridge-net-device.cc
  HEADER_FILES
    helper/bridge-helper.h
    model/bridge-channel.h
    model/bridge-net-device.h
  LIBRARIES_TO_LINK ${libnetwork}
  TEST_SOURCES ${test_sources}
)
