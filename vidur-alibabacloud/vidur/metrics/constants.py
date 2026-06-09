""" File to store names for different metrics captured """

import enum


class OperationMetrics(enum.Enum):
    MLP_UP_PROJ = "mlp_up_proj"
    MLP_ACTIVATION = "mlp_activation"
    MLP_DOWN_PROJ = "mlp_down_proj"
    MLP_DOWN_PROJ_ALL_REDUCE = "mlp_down_proj_all_reduce"
    ATTN_PRE_PROJ = "attn_pre_proj"
    ATTN_POST_PROJ = "attn_post_proj"
    ATTN_POST_PROJ_ALL_REDUCE = "attn_post_proj_all_reduce"
    ATTN_PREFILL = "attn_prefill"
    ATTN_KV_CACHE_SAVE = "attn_kv_cache_save"
    ATTN_DECODE = "attn_decode"
    ATTN_ROPE = "attn_rope"
    PIPELINE_SEND_RECV = "pipeline_send_recv"
    ADD = "add"
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"


class CpuOperationMetrics(enum.Enum):
    SCHEDULE = "schedule"
    SAMPLER_E2E = "sample_e2e"
    PREPARE_INPUTS_E2E = "prepare_inputs_e2e"
    MODEL_EXECUTION_E2E = "model_execution_e2e"
    PROCESS_MODEL_OUTPUTS = "process_model_outputs"
    RAY_COMM_TIME = "ray_comm_time"


class RequestMetricsTimeDistributions(enum.Enum):
    REQUEST_E2E_TIME = "request_e2e_time"
    REQUEST_E2E_TIME_NORMALIZED = "request_e2e_time_normalized"
    REQUEST_EXECUTION_TIME = "request_execution_time"
    REQUEST_EXECUTION_TIME_NORMALIZED = "request_execution_time_normalized"
    REQUEST_MODEL_EXECUTION_TIME = "request_model_execution_time"
    REQUEST_MODEL_EXECUTION_TIME_NORMALIZED = "request_model_execution_time_normalized"
    REQUEST_PREEMPTION_TIME = "request_preemption_time"
    REQUEST_SCHEDULING_DELAY = "request_scheduling_delay"
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME = "request_execution_plus_preemption_time"
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED = (
        "request_execution_plus_preemption_time_normalized"
    )
    PREFILL_TIME_E2E = "prefill_e2e_time"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION = "prefill_time_execution_plus_preemption"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "prefill_time_execution_plus_preemption_normalized"
    )
    DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "decode_time_execution_plus_preemption_normalized"
    )
    
    # >
    # ARRIVED_AT = ("_arrived_at")
    ARRIVED_AT = ("arrived_at")
    PREFILL_ARRIVED_AT = ("prefill_arrived_at")
    # SCHEDULED_AT = ("_scheduled_at")
    SCHEDULED_AT = ("scheduled_at")
    # PREFILL_COMPLETED_AT = ("_prefill_completed_at")
    PREFILL_COMPLETED_AT = ("prefill_completed_at")
    DECODE_ARRIVED_AT = ("decode_arrived_at")
    # COMPLETED_AT = ("_completed_at")
    COMPLETED_AT = ("completed_at")
    # PREEMPTED_TIME = ("_preempted_time")
    PREEMPTED_TIME = ("preempted_time")
    DECODE_TIME = ("decode_time")
    
    
    PREFILL_REPLICA_ID = ("prefill_replica_id")
    DECODE_REPLICA_ID = ("decode_replica_id")
    PD_P2P_COMM_SIZE = ("pd_p2p_comm_size")
    PD_P2P_COMM_TIME = ("pd_p2p_comm_time")
    PD_P2P_COMM_BANDWIDTH = ("pd_p2p_comm_bandwidth")
    PD_P2P_COMM_BYTES_PER_TOKEN = ("pd_p2p_bytes_per_token")
    PD_P2P_COMM_DTYPE = ("pd_p2p_comm_dtype")
    


class TokenMetricsTimeDistribution(enum.Enum):
    DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME = (
        "decode_token_execution_plus_preemption_time"
    )


class RequestMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_TOKENS = "request_num_tokens"
    REQUEST_PREFILL_TOKENS = "request_num_prefill_tokens"
    REQUEST_DECODE_TOKENS = "request_num_decode_tokens"
    REQUEST_PD_RATIO = "request_pd_ratio"
    REQUEST_NUM_RESTARTS = "request_num_restarts"


class BatchMetricsCountDistribution(enum.Enum):
    BATCH_NUM_TOKENS = "batch_num_tokens"
    BATCH_NUM_PREFILL_TOKENS = "batch_num_prefill_tokens"
    BATCH_NUM_DECODE_TOKENS = "batch_num_decode_tokens"
    BATCH_SIZE = "batch_size"


class BatchMetricsTimeDistribution(enum.Enum):
    BATCH_EXECUTION_TIME = "batch_execution_time"


class RequestCompletionMetricsTimeSeries(enum.Enum):
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_COMPLETION = "request_completion"


class TokenCompletionMetricsTimeSeries(enum.Enum):
    PREFILL_COMPLETIONS = "prefill_completion"
    DECODE_COMPLETIONS = "decode_completion"
