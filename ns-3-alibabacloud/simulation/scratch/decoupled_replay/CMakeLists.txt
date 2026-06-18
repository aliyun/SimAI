# Decoupled Replay Binary
# Links ONLY against ns3 libraries -- NO astra-sim dependency
#
# Verification: nm scratch_decoupled_replay | grep -i simai must return empty

# Use explicit add_executable instead of create_scratch() to avoid
# path resolution issues when called from a subdirectory CMakeLists.txt

add_executable(scratch_decoupled_replay
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
)

# Additional include directories for ns3 internal headers
target_include_directories(scratch_decoupled_replay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/point-to-point
    ${CMAKE_SOURCE_DIR}/src/applications
)

# Link against ns3 libraries (same as what create_scratch does for non-static builds)
target_link_libraries(scratch_decoupled_replay
    "${ns3-libs}"
    "${ns3-contrib-libs}"
    "${ns3-external-libs}"
)

# Do NOT link astra-sim libraries -- this is the core independence requirement
