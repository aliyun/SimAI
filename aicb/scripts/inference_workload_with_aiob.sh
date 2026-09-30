SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

model_size=deepseek-671B
config_file_path=
phase=decode
seq_length=1024
micro_batch=32
world_size=8
tensor_model_parallel_size=1
expert_model_parallel_size=8
pipeline_model_parallel=1
moe_enable=true
result_dir=results/workload/
aiob_enable=false
aiob_forward_loops=10

dpsk_default_path="$SCRIPT_DIR/inference_configs/deepseek_default.json"
qwen3_moe_default_path="$SCRIPT_DIR/inference_configs/qwen3_moe_default.json"
qwen3_next_default_path="$SCRIPT_DIR/inference_configs/qwen3_next_default.json"

usage() {
  cat <<-EOF
Usage: $0 [OPTIONS]

  Generate inference workload with AIOB support.

Options:
  -m, --model-size <SIZE>
      Model size to use.
      Possible values: {deepseek-671B, qwen3-235B, qwen3-next-80B}.
      (Default: $model_size)

  -c, --config <FILE>
      Path to a custom configuration file.
      (Default: None)

  -p, --phase <PHASE>
      Inference phase.
      Possible values: {prefill, decode}.
      (Default: $phase)

  -s, --seq-length <LENGTH>
      Sequence length for the model.
      (Default: $seq_length)

  -b, --micro-batch <SIZE>
      Micro batch size.
      (Default: $micro_batch)

  -w, --world-size <SIZE>
      Total number of GPUs (world size).
      (Default: $world_size)

  -t, --tp-size <SIZE>
      Tensor model parallel size.
      (Default: $tensor_model_parallel_size)

  -e, --ep-size <SIZE>
      Expert model parallel size (for MoE models).
      (Default: $expert_model_parallel_size)

  -l, --pp-size <SIZE>
      Pipeline model parallel size.
      (Default: $pipeline_model_parallel)

  -M, --moe-enable
      Enable MoE (Mixture of Experts) support.
      (This is a boolean flag. Default: $moe_enable)

  -r, --result-dir <DIR>
      Directory to save the results.
      (Default: "$result_dir")

  -a, --aiob-enable
      Enable AIOB (All-In-One Block) support.
      (This is a boolean flag. Default: $aiob_enable)

  -f, --aiob-loops <LOOPS>
      Number of forward loops for AIOB.
      (Default: $aiob_forward_loops)

  -h, --help
      Display this help message and exit.

Example:
  sh $0 -m deepseek-671B -p decode -s 1024 -b 32 --aiob-enable

EOF
  exit 1
}

while [ $# -gt 0 ]
do
  case $1 in
    -m|--model_size)
      model_size=$2; shift;;
    -c|--config)
      config_file_path=$2; shift;;
    -p|--phase)
      phase=$2; shift;;
    -s|--seq_length)
      seq_length=$2; shift;;
    -b|--micro_batch)
      micro_batch=$2; shift;;
    -w|--world_size)
      world_size=$2; shift;;
    -t|--tensor_model_parallel_size)
      tensor_model_parallel_size=$2; shift;;
    -e|--expert_model_parallel_size)
      expert_model_parallel_size=$2; shift;;
    -l|--pipeline_model_parallel)
      pipeline_model_parallel=$2; shift;;
    -M|--moe_enable)
      moe_enable=true;;
    -r|--result_dir)
      result_dir=$2; shift;;
    -a|--aiob_enable)
      aiob_enable=true;;
    -f|--aiob_forward_loops)
      aiob_forward_loops=$2; shift;;
    -h|--help)
      usage;;
    (*)
      break;;
  esac
  shift
done

case $model_size in
  deepseek-671B)
    model_name=DeepSeek-671B
    config_file_path=${config_file_path:-$dpsk_default_path}
    ;;
  qwen3-235B)
    model_name=Qwen3-Moe-235B
    config_file_path=${config_file_path:-$qwen3_moe_default_path}
    ;;
  qwen3-next-80B)
    model_name=Qwen3-Next-80B
    config_file_path=${config_file_path:-$qwen3_next_default_path}
    ;;
  (*)
    echo "Invalid model size: $model_size"
    usage;;
esac

# Build command with optional parameters
cmd="python -m workload_generator.SimAI_inference_workload_generator $model_name $config_file_path"

# Add optional parameters if they are set
if [ ! -z "$phase" ]; then
  cmd="$cmd --phase $phase"
fi

if [ ! -z "$seq_length" ]; then
  cmd="$cmd --seq_length $seq_length"
fi

if [ ! -z "$micro_batch" ]; then
  cmd="$cmd --micro_batch $micro_batch"
fi

if [ ! -z "$world_size" ]; then
  cmd="$cmd --world_size $world_size"
fi

if [ ! -z "$tensor_model_parallel_size" ]; then
  cmd="$cmd --tensor_model_parallel_size $tensor_model_parallel_size"
fi

if [ ! -z "$expert_model_parallel_size" ]; then
  cmd="$cmd --expert_model_parallel_size $expert_model_parallel_size"
fi

if [ ! -z "$pipeline_model_parallel" ]; then
  cmd="$cmd --pipeline_model_parallel $pipeline_model_parallel"
fi

if [ "$moe_enable" = true ]; then
  cmd="$cmd --moe_enable"
fi

if [ ! -z "$result_dir" ]; then
  cmd="$cmd --result_dir $result_dir"
fi

if [ "$aiob_enable" = true ]; then
  cmd="$cmd --aiob_enable"
fi

if [ ! -z "$aiob_forward_loops" ]; then
  cmd="$cmd --aiob_forward_loops $aiob_forward_loops"
fi

echo $cmd

$cmd
