#!/bin/bash
# set -e
# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo $SCRIPT_DIR

# Absolute paths to useful directories
GEM5_DIR="${SCRIPT_DIR:?}"/../../extern/network_backend/garnet/gem5_astra/
ASTRA_SIM_DIR="${SCRIPT_DIR:?}"/../../astra-sim
INPUT_DIR="${SCRIPT_DIR:?}"/../../inputs
NS3_DIR="${SCRIPT_DIR:?}"/../../../ns-3-alibabacloud
NS3_APPLICATION="${NS3_DIR:?}"/simulation/src/applications/
SIM_LOG_DIR=/etc/astra-sim
BUILD_DIR="${SCRIPT_DIR:?}"/build/
RESULT_DIR="${SCRIPT_DIR:?}"/result/
BINARY="${BUILD_DIR}"/gem5.opt
ASTRA_SIM_LIB_DIR="${SCRIPT_DIR:?}"/build/AstraSim

# Functions
function setup {
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${RESULT_DIR}"
}

function cleanup {
    echo $BUILD_DIR
    rm -rf "${BUILD_DIR}"
    rm -rf "${NS3_DIR}"/simulation/build
    rm -rf "${NS3_DIR}"/simulation/cmake-cache
    rm -rf "${NS3_APPLICATION}"/astra-sim 
    cd "${SCRIPT_DIR:?}"
}

function cleanup_result {
    rm -rf "${RESULT_DIR}"
}

function compile {
    # Only compile & Run the AstraSimNetwork ns3program
    if [ ! -f '"${INPUT_DIR}"/inputs/config/SimAI.conf' ]; then
        echo ""${INPUT_DIR}"/config/SimAI.conf is not exist"
        cp "${INPUT_DIR}"/config/SimAI.conf "${SIM_LOG_DIR}"/config/SimAI.conf
    fi
    cd "${NS3_DIR}/simulation"
    # CC='gcc-4.9' CXX='g++-4.9' 
    CC='gcc' CXX='g++'
    mkdir build
    cd build
    cmake ..
    make -j12 AstraSimNetworkNs3
    cd "${SCRIPT_DIR:?}"
}

# this cannot work; should use cmake to debug, neither waf nor ns3
function debug {
#    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/AstraSimNetwork.cc "${NS3_DIR}"/simulation/scratch/
#    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/*.h "${NS3_DIR}"/simulation/scratch/
    cd "${NS3_DIR}/simulation"
    CC='gcc-4.9' CXX='g++-4.9' 
    ./waf configure
    ./waf --run 'scratch/AstraSimNetwork' --command-template="gdb --args %s mix/config.txt"

    ./waf --run 'scratch/AstraSimNetwork mix/config.txt'

    cd "${SCRIPT_DIR:?}"
}

# Main Script
case "$1" in
-l|--clean)
    cleanup;;
-lr|--clean-result)
    cleanup
    cleanup_result;;
-d|--debug)
    setup
    debug;; # todo: remove?
-c|--compile)
    setup
    compile;;
-r|--run)
    setup
    compile;;
-h|--help|*)
    printf "Prints help message";;
esac