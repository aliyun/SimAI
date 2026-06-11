cmake_minimum_required(VERSION 3.15)

# Setup project
project(AstraSimNetwork)

# Compile AnalyticalBackend binary
add_subdirectory ("${PROJECT_SOURCE_DIR}/../../astra-sim/network_frontend/ns3" simai_ns3)
