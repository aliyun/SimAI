project(SimAI_analytical)

file(GLOB SOURCES CONFIGURE_DEPENDS "*.cc")
file(GLOB HEADERS CONFIGURE_DEPENDS "*.hh" "*.h")

add_executable(SimAI_analytical ${SOURCES} ${HEADERS})
target_include_directories(SimAI_analytical PRIVATE "${ASTRA_SIM_DIR}")
target_link_libraries(SimAI_analytical PRIVATE AstraSim)
