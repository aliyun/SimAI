#!/bin/sh

set -x
: ${WORLD_SIZE:=1}
: ${RANK:=0}
: ${MASTER_ADDR:="localhost"}
: ${MASTER_PORT:=29500}
NUM_GPUS=$(nvidia-smi -L | wc -l) # Get the number of GPUs on a single node
model_size=13
num_layers=40
num_attention_heads=40
hidden_size=5120
seq_length=2048
micro_batch=1
epoch_num=1
tensor_model_parallel_size=8
pipeline_model_parallel=1
vocab_size=50257
model_name=gpt_13b
ga_num=2
sp_enable=
frame=Megatron
aiob_enable=
max_position_embeddings=4096
num_experts=1
moe_enable=
enable_visual=
workload_only=
usage() {
  echo "Usage: \$0 [options]
    options:
      --frame              Communication framework: $frame
      --world_size              World size (number of nodes): $WORLD_SIZE
      --tensor_model_parallel_size                  Tensor parallelism size: $tensor_model_parallel_size
      --pipeline_model_parallel                  Pipeline parallelism size: $pipeline_model_parallel
      --global_batch            Global batch size: $global_batch
      --micro_batch             Micro batch size: $micro_batch
      --num_layers              Number of layers: $num_layers
      --seq_length              Sequence length: $seq_length
      --hidden_size             Hidden size: $hidden_size
      --epoch_num               Number of epochs: $epoch_num
      --num_attention_heads     Number of attention heads: $num_attention_heads
      --aiob_enable             Enable AIOB: $aiob_enable
      --enable_visual           Enable Visualization $enable_visual 
      --workload_only           generate workload only
      --use_flash_attn          Use flash attention: $use_flash_attn
      --swiglu                  Use SWIGLU: $swiglu
      --ffn_hidden_size         FFN hidden size: $ffn_hidden_size
      --comp_filepath           Computation file path: $comp_filepath
      --model_name              Model name: $model_name
      -m, --model_size          model size, defaults to $model_size (possible values: 175, 22, 13, 7)
      --max_position_embeddings Max position embeddings: $max_position_embeddings
      --nnodes                  Number of nodes: $WORLD_SIZE
      --node_rank               Rank of the node: $RANK
      --nproc_per_node          Number of GPUs per node: $NUM_GPUS
      --master_addr             Master address: $MASTER_ADDR
      --master_port             Master port: $MASTER_PORT
      --me_enable                enable moe
      --moe_router_topk         Number of experts to route to for each token.
      --expert_model_parallel_size     Degree of expert model parallelism
      --num_experts          Number of experts in the MoE model.  
      --moe_grouped_gemm        apply grouped gemm
      -h, --help                Display this help and exit"1>&2; exit 1;
}
while [ $# -gt 0 ]
do
echo "Processing argument: $1"
  case $1 in
    --frame)
      frame=$2; shift;;
    --world_size)
      world_size=$2; shift;;
    --tensor_model_parallel_size|tp_num)
      tensor_model_parallel_size=$2; shift;;
    --pipeline_model_parallel|pp_num)
      pipeline_model_parallel=$2; shift;;
    --global_batch)
      global_batch=$2; shift;;
    --micro_batch)
      micro_batch=$2; shift;;
    --num_layers)
      num_layers=$2; shift;;
    --seq_length)
      seq_length=$2; shift;;
    --hidden_size)
      hidden_size=$2; shift;;
    --epoch_num)
      epoch_num=$2; shift;;
    --num_attention_heads)
      num_attention_heads=$2; shift;;
    --aiob_enable)
      aiob_enable=--aiob_enable;;
    --enable_visual)
      enable_visual=--enable_visual;;
    --workload_only)
      workload_only=--workload_only;;
    --use_flash_attn)
      use_flash_attn=--use_flash_attn;;
    --swiglu)
      swiglu=--swiglu;;
    --ffn_hidden_size)
      ffn_hidden_size=$2; shift;;
    --sp|--sp-enable|--enable_sequence_parallel)
      sp_enable=--enable_sequence_parallel;;
    --comp_filepath)
      comp_filepath=$2; shift;;
    -m|--model_size)
      model_size=$2; shift;;
    --moe_enable)
      moe_enable=--moe_enable;;
    --moe_router_topk|--topk)
      moe_router_topk=$2; shift;;
    --num_experts|--experts)
      num_experts=$2; shift;;
    --expert_model_parallel_size|--ep)
      expert_model_parallel_size=$2; shift;;
    --grouped_gemm|--moe_grouped_gemm)
      grouped_gemm=--moe_grouped_gemm;;
    --nnodes)
      WORLD_SIZE=$2;shift;;
    --node_rank)
      RANK=$2;shift;;
    --nproc_per_node)
      NUM_GPUS=$2;shift;;
    --master_addr)
      MASTER_ADDR=$2;shift;;
    --master_port)
      MASTER_PORT=$2;shift;;  
    -h|--help)
      usage ;;
    (*)
      break;;
  esac

  shift
done

case $model_size in
  175)
    model_name=gpt_175B
    num_layers=96
    hidden_size=12288
    num_attention_heads=96
    tensor_model_parallel_size=8
    ;;
  22)
    model_name=gpt_22B
    num_layers=48
    hidden_size=6144
    num_attention_heads=64
    tensor_model_parallel_size=8
    ;;
  13)
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
  7)
    model_name=gpt_7B
    num_layers=36
    hidden_size=4096
    num_attention_heads=32
    ;;
  405)
    model_name=llama_405B
    num_layers=128
    hidden_size=16384
    ffn_hidden_size=53248
    num_attention_heads=128
    tensor_model_parallel_size=8
    pipeline_model_parallel=16
    ;;
  65)
    model_name=llama_65B
    num_layers=80
    hidden_size=8192
    ffn_hidden_size=28672
    num_attention_heads=64
    tensor_model_parallel_size=8
    pipeline_model_parallel=2
    ;;
  moe)
    model_name=Mixtral_8*7B
    num_layers=32
    hidden_size=4096
    num_attention_heads=32
    ffn_hidden_size=14336
    tensor_model_parallel_size=2
    moe_enable=--moe_enable
    grouped_gemm=--moe_grouped_gemm
    ;;
  deepseek671)
    model_name=DeepSeek_671B
    frame=DeepSeek
    moe_enable=--moe_enable
    num_layers=61
    hidden_size=18432
    num_attention_heads=128
    ffn_hidden_size=2048
    q_lora_rank=1536
    kv_lora_rank=512
    qk_nope_dim=128
    qk_rope_dim=64
    v_head_dim=128
    num_experts=256
    n_shared_expert=1
    n_dense_layer=3
    tensor_model_parallel_size=1
    ;;
  deepseek236)
    model_name=DeepSeek_236B
    frame=DeepSeek
    moe_enable=--moe_enable
    num_layers=60
    hidden_size=12288
    num_attention_heads=128
    ffn_hidden_size=1536
    q_lora_rank=1536
    kv_lora_rank=512
    qk_nope_dim=128
    qk_rope_dim=64
    v_head_dim=128
    num_experts=160
    n_shared_expert=2
    n_dense_layer=1
    tensor_model_parallel_size=1
    ;;
  deepseek16)
    model_name=DeepSeek_16B
    frame=DeepSeek
    moe_enable=--moe_enable
    num_layers=27
    hidden_size=10944
    num_attention_heads=16
    ffn_hidden_size=1408
    q_lora_rank=0
    kv_lora_rank=512
    qk_nope_dim=128
    qk_rope_dim=64
    v_head_dim=128
    num_experts=64
    n_shared_expert=2
    n_dense_layer=1
    tensor_model_parallel_size=1
    ;;
  deepseek)
    frame=DeepSeek
    model_name="DeepSeek"
    moe_enable=--moe_enable
    ;;
    # set everything else from cmdline
    # or it defaults to utils/utils.py's arg parser
  (*)
    echo "Only support model size 405,175,22,13,7 or moe; using default size 13"
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
esac

dp_num=$((world_size/tensor_model_parallel_size/pipeline_model_parallel))
global_batch=$((ga_num*dp_num*micro_batch))
if [ $workload_only ]; then
  script="python -m workload_generator.generate_megatron_workload" 
else
  script="./aicb.py"
fi

cmd="$script \
  --frame=$frame \
  --model_name=$model_name \
  --world_size=$(($WORLD_SIZE * $NUM_GPUS)) \
  --tensor_model_parallel_size=$tensor_model_parallel_size \
  --micro_batch=$micro_batch \
  --global_batch=$global_batch \
  --epoch_num=$epoch_num \
  --num_layers=$num_layers \
  --hidden_size=$hidden_size \
  --num_attention_heads=$num_attention_heads \
  --seq_length=$seq_length \
  --vocab_size=$vocab_size \
  --pipeline_model_parallel=$pipeline_model_parallel \
  --use-distributed-optimizer \
  --max_position_embeddings=$max_position_embeddings \
  ${aiob_enable} \
  ${enable_visual} \
  ${workload_only} \
  ${sp_enable} \
  ${use_flash_attn} \
  ${swiglu} \
  ${ffn_hidden_size:+--ffn_hidden_size=$ffn_hidden_size} \
  ${comp_filepath:+--comp_filepath=$comp_filepath} \
  ${moe_enable} \
  ${moe_router_topk:+--moe_router_topk=$moe_router_topk} \
  ${num_experts:+--num_experts=$num_experts} \
  ${expert_model_parallel_size:+--expert_model_parallel_size=$expert_model_parallel_size} \
  ${grouped_gemm} \
  ${q_lora_rank:+--q_lora_rank=$q_lora_rank} \
  ${kv_lora_rank:+--kv_lora_rank=$kv_lora_rank} \
  ${qk_nope_dim:+--qk_nope_dim=$qk_nope_dim} \
  ${qk_rope_dim:+--qk_rope_dim=$qk_rope_dim} \
  ${v_head_dim:+--v_head_dim=$v_head_dim} \
  ${n_shared_expert:+--n_shared_expert=$n_shared_expert} \
  ${n_dense_layer:+--n_dense_layer=$n_dense_layer} \
  "
echo $cmd

if [ $workload_only ]; then
  $cmd
else
  torchrun \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --nproc_per_node $NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $cmd
fi
