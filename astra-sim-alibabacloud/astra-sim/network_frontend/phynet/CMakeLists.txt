project(SimAI_phynet)

file(GLOB SOURCES CONFIGURE_DEPENDS "*.cc")
file(GLOB HEADERS CONFIGURE_DEPENDS "*.hh" "*.h")

add_executable(SimAI_phynet ${SOURCES} ${HEADERS})
target_include_directories(SimAI_phynet PRIVATE "${ASTRA_SIM_DIR}")
target_link_options(SimAI_phynet PRIVATE "-static-libstdc++")
target_compile_definitions(SimAI_phynet PRIVATE PHY_MTP)

if(USE_RDMA)
    target_compile_definitions(SimAI_phynet PRIVATE PHY_RDMA)
    target_link_libraries(SimAI_phynet PRIVATE AstraSim ibverbs pthread)
else()
    target_link_libraries(SimAI_phynet PRIVATE AstraSim)
endif()
