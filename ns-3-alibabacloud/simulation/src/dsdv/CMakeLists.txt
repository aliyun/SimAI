build_lib(
  LIBNAME dsdv
  SOURCE_FILES
    helper/dsdv-helper.cc
    model/dsdv-packet-queue.cc
    model/dsdv-packet.cc
    model/dsdv-routing-protocol.cc
    model/dsdv-rtable.cc
  HEADER_FILES
    helper/dsdv-helper.h
    model/dsdv-packet-queue.h
    model/dsdv-packet.h
    model/dsdv-routing-protocol.h
    model/dsdv-rtable.h
  LIBRARIES_TO_LINK ${libinternet}
  TEST_SOURCES test/dsdv-testcase.cc
)
