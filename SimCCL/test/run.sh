#!/bin/bash
 
if [ "$1" == "gdb" ]; then
  CMD="gdb"
  CMD=""
fi
 
export CUDA_VISIBLE_DEVICES=
# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=32

# Obtain the absolute path of script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
LD_LIBRARY_PATH="$SCRIPT_DIR/../build/lib/" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=MOCK_ROOT,MOCK \
       NCCL_CROSS_NIC=1 NCCL_TOPO_FILE=$SCRIPT_DIR/topo_8A100_4CX6.xml NCCL_NVLS_ENABLE=0 \
       NCCL_NUM_MOCK_GPU=64 NCCL_NUM_MOCK_NODE=8 $CMD ./MultiDevOneP $CORE_FILE
       