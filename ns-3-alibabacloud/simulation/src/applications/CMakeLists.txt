file(GLOB ASTRA_SIM_SOURCE_FILES 
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.cc")

list(FILTER ASTRA_SIM_SOURCE_FILES EXCLUDE  REGEX ".*BootStrapnet\\.cc")
list(FILTER ASTRA_SIM_SOURCE_FILES EXCLUDE  REGEX ".*SimAiFlowModelRdma\\.cc")
list(FILTER ASTRA_SIM_SOURCE_FILES EXCLUDE  REGEX ".*PhyMultiThread\\.cc")


include_directories($<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

build_lib(
  LIBNAME applications
  SOURCE_FILES
    helper/bulk-send-helper.cc
    helper/on-off-helper.cc
    helper/packet-sink-helper.cc
    helper/three-gpp-http-helper.cc
    helper/udp-client-server-helper.cc
    helper/udp-echo-helper.cc
    helper/rdma-client-helper.cc
    model/application-packet-probe.cc
    model/bulk-send-application.cc
    model/onoff-application.cc
    model/packet-loss-counter.cc
    model/packet-sink.cc
    model/seq-ts-echo-header.cc
    model/seq-ts-header.cc
    model/seq-ts-size-header.cc
    model/three-gpp-http-client.cc
    model/three-gpp-http-header.cc
    model/three-gpp-http-server.cc
    model/three-gpp-http-variables.cc
    model/udp-client.cc
    model/udp-echo-client.cc
    model/udp-echo-server.cc
    model/udp-server.cc
    model/udp-trace-client.cc
    model/rdma-client.cc
    model/simple-seq-ts-header.cc
    ${ASTRA_SIM_SOURCE_FILES}
  HEADER_FILES
    helper/bulk-send-helper.h
    helper/on-off-helper.h
    helper/packet-sink-helper.h
    helper/three-gpp-http-helper.h
    helper/udp-client-server-helper.h
    helper/udp-echo-helper.h
    helper/rdma-client-helper.h
    model/application-packet-probe.h
    model/bulk-send-application.h
    model/onoff-application.h
    model/packet-loss-counter.h
    model/packet-sink.h
    model/seq-ts-echo-header.h
    model/seq-ts-header.h
    model/seq-ts-size-header.h
    model/three-gpp-http-client.h
    model/three-gpp-http-header.h
    model/three-gpp-http-server.h
    model/three-gpp-http-variables.h
    model/udp-client.h
    model/udp-echo-client.h
    model/udp-echo-server.h
    model/udp-server.h
    model/udp-trace-client.h
    model/rdma-client.h
    model/simple-seq-ts-header.h
    # ${ASTRA_SIM_HEADER_FILES}
  LIBRARIES_TO_LINK ${libinternet}
                    ${libstats}
  TEST_SOURCES
    test/three-gpp-http-client-server-test.cc
    test/bulk-send-application-test-suite.cc
    test/udp-client-server-test.cc
)
