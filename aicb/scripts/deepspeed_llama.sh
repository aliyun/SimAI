#!/bin/sh


set -x
: ${WORLD_SIZE:=1}
: ${RANK:=0}
: ${MASTER_ADDR:="localhost"}
: ${MASTER_PORT:=29500}
model_name=llama_7b
zero_stage=3
model_size=7
num_layers=32
epoch_num=10
num_attention_heads=32
hidden_size=4096
ffn_hidden_size=11008
reduce_bucket_size=500000000
allgather_bucket_size=500000000
prefetch_bucket_size=1000000000
max_live_parameters=1000000000
param_persistence_threshold=100000
seq_len=2048
batch_size=4
contiguous_gradients=
aiob_enable=
enable_visual=
workload_only=

usage() {
  echo "Usage: $0 [options]
    options:
      --model_name              model_name: $model_name
      --zero_stage              zero_stage: $zero_stage
      --epoch_num               num of iterations: $epoch_num
      --batch_size              micro batch_size: $batch_size
      --enable_visual           enable visual html output files
      --workload_only           generate workload only
      -m, --model-size          llama model size.(7/13/30/65): $model_size
      --reduce-bucket-size      size of reduce bucket: $reduce_bucket_size
      --allgather-bucket-size   size of all_gather bucket(only used in stage1,2): $reduce_bucket_size
      --prefetch-bucket-size    size of all_gather prefetch bucket(only used in stage3): $prefetch_bucket_size 
      --max-live-parameters     max size of params that have been all_gather(only used in stage3): $max_live_parameters 
      --param-persistence-threshold    threshold of param that is all-gather before forward(only used in stage3): $param_persistence_threshold 
      --seq-len                 seq-len: $seq_len
      --contiguous-gradients    use reduce instead of all_reduce (only used in stage2)
      -h, --help" 1>&2; exit 1;
}

while [ $# -gt 0 ]
do
echo "Processing argument: $1"
  case $1 in
    --model_name|--model-name)
      model_name=$2 ; shift;;
    --stage|--zero-stage|--zero_stage)
      zero_stage=$2 ; shift;;
    --epoch-num|--epoch_num)
      epoch_num=$2 ; shift;;
    --batch-size|--micro_batch|--batch_size)
      batch_size=$2 ; shift;;
    -m|--model-size)
      model_size=$2 ; shift;;
    --reduce-bucket-size|--reduce_bucket_size)
      reduce_bucket_size=$2 ; shift;;
    --param-persistence-threshold|--param_persistence_threshold)
      param_persistence_threshold=$2 ; shift;;
    --max-live-parameters|--max_live_parameters)
      prefetch_bucket_size=$2 ; shift;;
    --allgather-bucket-size|--allgather_bucket_size)
      allgather_bucket_size=$2 ; shift;;
    --seq-len|--seq_len)
      seq_len=$2 ; shift;;
    --aiob_enable)
      aiob_enable=--aiob_enable;;
    --enable_visual)
      enable_visual=--enable_visual;;
    --workload_only)
      workload_only=--workload_only;;
    --contiguous-gradients|--contiguous_gradients)
      contiguous_gradients=--contiguous_gradients; shift;;
    -h|--help)
      usage ;;
    (*)
      break;;
  esac
  # Fetch next argument as 1st
  shift
done

case $model_size in
  13)
    model_name=llama_13b hidden_size=5120; ffn_hidden_size=13824; num_layers=40; num_attention_heads=40; shift;;
  30)
    model_name=llama_30b hidden_size=6656; ffn_hidden_size=17920; num_layers=60; num_attention_heads=52; shift;;
  65)
    model_name=llama_65b hidden_size=8192; ffn_hidden_size=22016; num_layers=80; num_attention_heads=64; shift;;
  7)
    ;;
  (*)
    echo "only suport model size 7b, 13b, 30b, 65b, got $model_size";;
esac

script="./aicb.py"

torchrun \
--nnodes ${WORLD_SIZE} \
--node_rank $RANK \
--nproc_per_node gpu \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
$script --frame=DeepSpeed --model_name=$model_name --stage=$zero_stage --world_size=$((WORLD_SIZE*8)) \
  --micro_batch=$batch_size --global_batch=$((WORLD_SIZE*8*batch_size))  --epoch_num=$epoch_num \
  --num_layers=$num_layers --hidden_size=$hidden_size --ffn_hidden_size=$ffn_hidden_size --num_attention_heads=$num_attention_heads \
  --reduce_bucket_size=$reduce_bucket_size --allgather_bucket_size=$allgather_bucket_size --seq_len=$seq_len \
  --max_live_parameters=$max_live_parameters --param_persistence_threshold=$param_persistence_threshold $contiguous_gradients $aiob_enable $enable_visual $workload_only