cmake_minimum_required(VERSION 3.15)

# Setup project
project (AstraSim_Analytical)

# Compile AstraSim library
add_subdirectory("${PROJECT_SOURCE_DIR}/../../" AstraSim)

# Compile AnalyticalBackend binary
add_subdirectory ("${PROJECT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical" simai_analytical)
