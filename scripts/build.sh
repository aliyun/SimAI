SCRIPT_DIR=$(dirname "$(realpath $0)")
ROOT_DIR=$(realpath "${SCRIPT_DIR:?}"/..)
NS3_DIR="${ROOT_DIR:?}"/ns-3-alibabacloud
SIMAI_DIR="${ROOT_DIR:?}"/astra-sim-alibabacloud
SOURCE_NS3_BIN_DIR="${ROOT_DIR:?}"/ns-3-alibabacloud/simulation/build/AstraSimNetworkNs3
SOURCE_ANA_BIN_DIR="${SIMAI_DIR:?}"/build/simai_analytical/build/simai_analytical/SimAI_analytical
SOURCE_PHY_BIN_DIR="${SIMAI_DIR:?}"/build/simai_phy/build/simai_phynet/SimAI_phynet

TARGET_BIN_DIR="${SCRIPT_DIR:?}"/../bin
function compile {
    local option="$1" 
    case "$option" in
    "ns3")
        mkdir -p "${TARGET_BIN_DIR:?}"
#        rm -rf "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/
#        mkdir -p "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface
#        cp -r "${NS3_DIR:?}"/* "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr ns3
        ./build.sh -c ns3    
        ln -s "${SOURCE_NS3_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_simulator;;
    "phy")
        mkdir -p "${TARGET_BIN_DIR:?}"
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy
        ./build.sh -c phy 
        ln -s "${SOURCE_PHY_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_phynet;;
    "analytical")
        mkdir -p "${TARGET_BIN_DIR:?}"
        mkdir -p "${ROOT_DIR:?}"/results
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical
        ./build.sh -c analytical 
        ln -s "${SOURCE_ANA_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_analytical;;
    esac
}

function cleanup_build {
    local option="$1"
    case "$option" in
    "ns3")
        rm  "${TARGET_BIN_DIR:?}"/SimAI_simulator
#        rm -rf "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr ns3;;
    "phy")
        rm  "${TARGET_BIN_DIR:?}"/SimAI_phynet
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy;;
    "analytical")
        rm  "${TARGET_BIN_DIR:?}"/SimAI_analytical
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical;;
    esac
}

# Main Script
case "$1" in
-l|--clean)
    cleanup_build "$2";;
-c|--compile)
    compile "$2";;
-h|--help|*)
    printf -- "help message\n"
    printf -- "-c|--compile mode supported ns3/phy/analytical  (example:./build.sh -c ns3)\n"
    printf -- "-l|--clean  (example:./build.sh -l ns3)\n"
    printf -- "-lr|--clean-result mode  (example:./build.sh -lr ns3)\n"
esac