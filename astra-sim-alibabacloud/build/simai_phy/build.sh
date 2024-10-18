#!/bin/bash

# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BUILD_DIR="${SCRIPT_DIR:?}"/build/
RESULT_DIR="${SCRIPT_DIR:?}"/result/
BIN_DIR="${BUILD_DIR}"/SimAiPhy/bin/

# Functions
function cleanup_build {
    rm -rf "${BUILD_DIR}"
}

function cleanup_result {
    rm -rf "${RESULT_DIR}"
}

function setup {
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${RESULT_DIR}"
}

function compile {
    local option="$1" 
    cd "${BUILD_DIR}" || exit
    case "$option" in
    "RDMA")
        cmake -DUSE_RDMA=TRUE ..
        make;;
    esac
}


# Main Script
case "$1" in
-l|--clean)
    cleanup_build;;
-lr|--clean-result)
    cleanup_build
    cleanup_result;;
-c|--compile)
    setup
    compile "$2";;
-h|--help|*)
    echo "AnalyticalAstra build script."
    echo "Run $0 -c to compile.";;
esac

