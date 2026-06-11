cmake_minimum_required(VERSION 3.15)

project (SimAi_Phy)

add_subdirectory("${PROJECT_SOURCE_DIR}/../../" AstraSim)

add_subdirectory ("${PROJECT_SOURCE_DIR}/../../astra-sim/network_frontend/phynet" simai_phynet)
