#!/bin/bash
# set -e
# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo $SCRIPT_DIR

# Absolute paths to useful directories
GEM5_DIR="${SCRIPT_DIR:?}"/../../extern/network_backend/garnet/gem5_astra/
ASTRA_SIM_DIR="${SCRIPT_DIR:?}"/../../astra-sim
INPUT_DIR="${SCRIPT_DIR:?}"/../../inputs
NS3_DIR="${SCRIPT_DIR:?}"/../../extern/network_backend/ns3-interface
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
    rm -rf "${NS3_APPLICATION}"/SimCCL
    cd "${SCRIPT_DIR:?}"
}

function cleanup_result {
    rm -rf "${RESULT_DIR}"
}

function compile_astrasim {
    cd "${BUILD_DIR}" || exit
    cmake ..
    make
}

function compile {
    # Only compile & Run the AstraSimNetwork ns3program
    # if [ ! -f '"${INPUT_DIR}"/inputs/config/SimAI.conf' ]; then
    #     echo ""${INPUT_DIR}"/config/SimAI.conf is not exist"
    #     cp "${INPUT_DIR}"/config/SimAI.conf "${SIM_LOG_DIR}"/config/SimAI.conf
    # fi
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/AstraSimNetwork.cc "${NS3_DIR}"/simulation/scratch/
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/*.h "${NS3_DIR}"/simulation/scratch/
    rm -rf "${NS3_APPLICATION}"/astra-sim 
    cp -r "${ASTRA_SIM_DIR}" "${NS3_APPLICATION}"/
    # Copy versioned MockNccl translation layer into the ns3 app tree (flat layout).
    # SIMAI_NCCL_VERSION selects which mock version to build (default: v2.30).
    MOCK_VERSION="${SIMAI_NCCL_VERSION:-v2.30}"
    MOCK_SRC="${SCRIPT_DIR}/../../../SimCCL/src/mock/${MOCK_VERSION}"
    if [ ! -d "$MOCK_SRC" ]; then
        echo "[ERROR] SimCCL mock version '${MOCK_VERSION}' not found at: ${MOCK_SRC}" >&2
        echo "Supported versions: v2.20, v2.30" >&2
        exit 1
    fi
    rm -rf "${NS3_APPLICATION}"/SimCCL
    mkdir -p "${NS3_APPLICATION}"/SimCCL/mock
    cp -r "$MOCK_SRC"/* "${NS3_APPLICATION}"/SimCCL/mock/
    cd "${NS3_DIR}/simulation"
    CC='gcc' CXX='g++' 
    ./ns3 configure -d optimized --enable-mtp -- -DCMAKE_CXX_FLAGS="-g -O2"
    ./ns3 build

    cd "${SCRIPT_DIR:?}"
}

function debug {
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/AstraSimNetwork.cc "${NS3_DIR}"/simulation/scratch/
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/*.h "${NS3_DIR}"/simulation/scratch/
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
    debug;;
-c|--compile)
    setup
    compile_astrasim
    compile;;
-r|--run)
    setup
    compile;;
-h|--help|*)
    printf "Prints help message";;
esac