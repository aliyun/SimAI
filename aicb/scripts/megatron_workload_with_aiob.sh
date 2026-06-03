#!/bin/sh


frame=Megatron
world_size=32
tensor_model_parallel_size=8
pipeline_model_parallel=1
global_batch=1024
micro_batch=1
num_layers=40
seq_length=4096
hidden_size=5120
epoch_num=1
num_attention_heads=40
aiob_enable=
use_flash_attn=
swiglu=
sp_enable=
ffn_hidden_size=
comp_filepath=
model_size=13
max_position_embeddings=4096
vocab_size=50257
num_experts=1
moe_enable=
recompute_activations=
gpu_type=None
usage() {
  echo "Usage: \$0 [options]
    options:
      --frame              communication framework, defaults to $frame
      --world_size              world size, defaults to $world_size
      --tensor_model_parallel_size                  tensor parallelism size, defaults to $tensor_model_parallel_size
      --pipeline_model_parallel                  pipeline parallelism size, defaults to $pipeline_model_parallel
      --global_batch            global batch size, defaults to $global_batch
      --micro_batch             micro batch size, defaults to $micro_batch
      --num_layers              number of layers, defaults to $num_layers
      --seq_length              sequence length, defaults to $seq_length
      --hidden_size             hidden size, defaults to $hidden_size
      --epoch_num               number of epochs, defaults to $epoch_num
      --use_distributed_optimizer use distributed optimizer
      --num_attention_heads     number of attention heads, defaults to $num_attention_heads
      --aiob_enable             enable AIOB
      --use_flash_attn          use flash attention
      --swiglu                  use swiglu
      --ffn_hidden_size         FFN hidden size
      --comp_filepath           computation file path
      --max_position_embeddings max position embeddings, defaults to $max_position_embeddings
      -m, --model_size          model size, defaults to $model_size (possible values: 175, 22, 13, 7, moe, deepseek671, deepseek236, deepseek16)
      --moe_enable             enable moe
      --moe_router_topk         Number of experts to route to for each token.
      --expert_model_parallel_size     Degree of expert model parallelism
      --num_experts          Number of experts in the MoE model.  
      --moe_grouped_gemm        apply grouped gemm
      -h, --help                display this help and exit" 1>&2; exit 1;
}


while [ $# -gt 0 ]
do
   
  case $1 in
    --gpu_type)
      gpu_type=$2; shift;;
    --frame)
      frame=$2; shift;;
    --world_size)
      world_size=$2; shift;;
    --tensor_model_parallel_size|--tp)
      tensor_model_parallel_size=$2; shift;;
    --pipeline_model_parallel|--pp)
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
    --aiob_enable|--aiob)
      aiob_enable=--aiob_enable;;
    --use_flash_attn|--flash_attn)
      use_flash_attn=--use_flash_attn;;
    --swiglu)
      swiglu=--swiglu;;
    --ffn_hidden_size)
      ffn_hidden_size=$2; shift;;
    --sp|--sp-enable)
      sp_enable=--enable_sequence_parallel;;
    --comp_filepath)
      comp_filepath=$2; shift;;
    -m|--model_size)
      model_size=$2; shift;;
    --max_position_embeddings)
      max_position_embeddings=$2; shift;;
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
    --recompute_activations|--recompute)
      recompute_activations=--recompute_activations;;
    -h|--help)
      usage;;
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
    tensor_model_parallel_size=4
    ;;
  405)
    model_name=llama_405B
    num_layers=128
    hidden_size=16384
    ffn_hidden_size=53248
    num_attention_heads=128
    ;;
  moe)
    model_name=Mixtral_8*7B
    num_layers=32
    hidden_size=4096
    num_attention_heads=32
    ffn_hidden_size=14336
    tensor_model_parallel_size=4
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
    echo "Only support model size 175, 22,13 or 7; using default size 13"
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
esac


cmd="python -m workload_generator.SimAI_training_workload_generator \
  --gpu_type=$gpu_type \
  --frame=$frame \
  --world_size=$world_size \
  --tensor_model_parallel_size=$tensor_model_parallel_size \
  --pipeline_model_parallel=$pipeline_model_parallel \
  --global_batch=$global_batch \
  --micro_batch=$micro_batch \
  --num_layers=$num_layers \
  --seq_length=$seq_length \
  --hidden_size=$hidden_size \
  --epoch_num=$epoch_num \
  --num_attention_heads=$num_attention_heads \
  --model_name=$model_name \
  --max_position_embeddings=$max_position_embeddings \
  --vocab_size=$vocab_size \
  --use-distributed-optimizer
  ${aiob_enable} \
  ${use_flash_attn} \
  ${swiglu} \
  ${sp_enable} \
  ${recompute_activations} \
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
  " \

echo $cmd


$cmd
