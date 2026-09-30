#!/bin/sh

set -x

begin_size=4096
end_size=8589934592
epoch_num=1
iter_num=500
test_comm=all_reduce
frame=collective_test
model_name=all_reduce
multi_all_reduce_enable=0

usage() {
  echo "Usage: $0 [options]
    options:
      --iter_num                num of iterations: $iter_num
      --begin_size              start message size of test: $begin_size
      --end_size                end message size of test: $end_size
      --test_comm               collective communication type: $test_comm
      --multi_all_reduce_enable  enable muti all_reduce opration: $multi_all_reduce_enable
      -h, --help" 1>&2; exit 1;
}

while [ $# -gt 0 ]
do
  case $1 in
    --model_name|--model-name)
        model_name=$2; shift;;
    --iter-num|--iter_num)
      iter_num=$2 ; shift;;
    --begin_size|--begin-size)
      begin_size=$2 ; shift;;
    --end_size|--end-size)
      end_size=$2 ; shift;;
    --test_comm|--test-comm)
      test_comm=$2 ; shift;;
    --multi_all_reduce_enable|--muti-all-reduce-enable)
      multi_all_reduce_enable=$2 ; shift;;
    -h|--help)
      usage ;;
    (*)
      break;;
  esac
  # Fetch next argument as 1st
  shift
done


script="./aicb.py"

if [ "$multi_all_reduce_enable" -eq 0 ]; then
    echo  "torchrun \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --nproc_per_node gpu \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $script --iter_num=$iter_num --world_size=$((WORLD_SIZE*8))\
    --begin_size=$begin_size --end_size=$end_size --test_comm=$test_comm --model_name=$model_name\
    --frame=standard_check --multi_all_reduce_enable=$multi_all_reduce_enable"
else
   echo  "torchrun \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --nproc_per_node gpu \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $script --iter_num=$iter_num --world_size=$((WORLD_SIZE*8))\
    --begin_size=$begin_size --end_size=$end_size --test_comm=$test_comm --model_name=$model_name\
    --frame=standard_check --multi_all_reduce_enable=$multi_all_reduce_enable --pipeline_model_parallel=$WORLD_SIZE"
fi 

if [ "$multi_all_reduce_enable" -eq 0 ]; then
    torchrun \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --nproc_per_node gpu \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $script --iter_num=$iter_num --world_size=$((WORLD_SIZE*8))\
    --begin_size=$begin_size --end_size=$end_size --test_comm=$test_comm --model_name=$model_name\
    --frame=collective_test --multi_all_reduce_enable=$multi_all_reduce_enable 
else
    torchrun \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --nproc_per_node gpu \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $script --iter_num=$iter_num --world_size=$((WORLD_SIZE*8))\
    --begin_size=$begin_size --end_size=$end_size --test_comm=$test_comm --model_name=$model_name\
    --frame=collective_test --multi_all_reduce_enable=$multi_all_reduce_enable --pipeline_model_parallel=$WORLD_SIZE
fi