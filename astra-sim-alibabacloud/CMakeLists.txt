set(use_rdma ${USE_RDMA})
set(use_analytical ${USE_ANALYTICAL})
file(GLOB astra_SRC 
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/collective/*.cc"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/fast-backend/*.cc"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/memory/*.cc"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/scheduling/*.cc"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/topology/*.cc"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/*.cc"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/*.cc"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/workload/*.cc"
    )
file(GLOB HEADERS
    "${PROJECT_SOURCE_DIR}/../../astra-sim/system/collective/*.hh"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/fast-backend/*.hh"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/memory/*.hh"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/scheduling/*.hh"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/topology/*.hh"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/*.hh"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/system/*.hh"
	"${PROJECT_SOURCE_DIR}/../../astra-sim/workload/*.hh"
)
if(use_rdma)
	add_definitions(-DPHY_RDMA)
	include_directories("$ENV{MPI_INCLUDE_PATH}")
	add_definitions(-DPHY_MTP)
	set(CMAKE_BUILD_TYPE Debug)
elseif(use_analytical)
	list(FILTER HEADERS EXCLUDE  REGEX ".*SimAiFlowModelRdma.hh")
	list(FILTER astra_SRC EXCLUDE REGEX ".*SimAiFlowModelRdma.cc")
	list(FILTER HEADERS EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/../../astra-sim/system/BootStrapnet.hh")
	list(FILTER astra_SRC EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/../../astra-sim/system/BootStrapnet.cc")
	list(FILTER HEADERS EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/../../astra-sim/system/PhyMultiThread.hh")
	list(FILTER astra_SRC EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/../../astra-sim/system/PhyMultiThread.cc")
	add_definitions(-DANALYTI)
endif()
include_directories("${PROJECT_SOURCE_DIR}/../../")
add_library(AstraSim ${astra_SRC})
set_property(TARGET AstraSim PROPERTY CXX_STANDARD 11)

