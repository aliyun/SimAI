# Understanding metrics logged by the simulator

## Preliminaries

For every request, we define the following key metrics:

1. Request arrival time ($a_r$): the time at which a request enters the system
2. Request schedule time ($s_r$): the time at which a given request is scheduled for the first time (irrespective of subsequent restarts).
3. Request completion time ($c_r$): the time at which a request completes.
4. Request prefill completion time ($f_r$): the time at which prefill completes and first output token is produced.
5. Request execution time ($e_r$): the total amount of time a request spends actually executing on GPUs (across all attempts) - excluding the time request is allocated on a replica but not executing due to pipeline-bubbles etc.
6. Request preemption time ($p_r$): the total amount of time a request spends request is allocated on a replica but not executing due to pipeline-bubbles, scheduling preemptions, time between restarts, etc (aggregate across all attempts).
7. Request scheduling delay ($d_r$): the total amount for which the request is waiting before getting scheduled ($s_r - a_r$).

Note that arrival, schedule and completion time refer to a specific point in time, where as, execution, preemption time, scheduling delay refer to period of time.

## Logged Metics

1. `request_inter_arrival_delay_histogram`: Histogram of difference between arrival times of adjacent requests ($a_{r+1} - a_r$).
2. `request_num_tokens_histogram`: Histogram of number of tokens (prefill + decode) across all requests.
3. `request_num_restarts_histogram`: Histogram of number of restarts for a given request. Note that this is expected to be a non-zero entity only when using vLLM or dSararthi schedulers - which restart requests in case a replica runs out of memory.
4. `request_e2e_time_cdf`: CDF of end-to-end request latency ($c_r - a_r$).
5. `request_e2e_time_normalised_cdf`: CDF of end-to-end request latency normalised by number of output tokens.
6. `request_execution_plus_preemption_times_cdf`: CDF of total time a request spends in the system excluding initial scheduling delay ($c_r - s_r$).
7. `request_scheduling_delay_cdf`: CDF of request scheduling delay ($s_r - a_r$).
8. `request_execution_time_cdf`: CDF of request execution time ($e_r$).
9. `request_preempted_time_cdf`: CDF of request preemption time ($p_r$).
10. `decode_token_execution_plus_preemption_times`: CDF of per decode token execution time and preemption time - i.e. inter-token delay observed by the user.
11. `batch_num_tokens_cdf`: CDF of total number of tokens to be processed in a batch (sum of prefill tokens + one per decode request). This distribution is useful towards understanding how the compute load is distributed across batches. Note that with iteration level scheduling a batch is formed at every iteration.
12. `batch_sizes_cdf`: CDF of batch sizes - usually larger batch sizes imply higher throughput.
13. `prefill_time_e2e_cdf`: CDF of end-to-end latency to the first output token (time-to-first-byte), i.e, time elapsed since the request arrival to the point where first output is generated ($f_r - a_r$).
14. `prefill_time_execution_plus_preemption_cdf`: CDF of total prefill process time excluding the initial scheduling delay ($f_r - s_r$). This metric is useful for tracking the prefill efficiency.
15. `prefill_time_execution_plus_preemption_normalized_cdf`: Similar to `prefill_time_execution_plus_preemption_cdf`, but normalized by the number of prefill tokens. This provides distribution independent of request prefill length, and thus, easier to analyze.
16. `decode_time_execution_plus_preemption_normalized_cdf`: CDF of total time spent processing decodes ($c_r - f_r$) normalized by the number of decode tokens. This provides an indicator similar to `decode_token_execution_plus_preemption_times`, however, this metric is presents an averaged over all decode tokens in the request.
17. `request_completions_time_series`: Time series of request completion times - this provides an indicator for makespan and helps in identifying the request processing rate (requests per second) by analyzing the slope of the curve.
18. `prefill_completions_time_series`: Time series of prefill token completion times - helps in identifying the prefill processing rate (prefill tokens per second) by analyzing the slope of the curve.
19. `decode_completions_time_series`: Time series of decode  completion times - helps in identifying the decode processing rate (decode tokens per second) by analyzing the slope of the curve.
20. `replica_{replica_id}_memory_usage_weighted_mean`: Memory usage statistics per replica-level - tracks the mean utilization value across entire execution time.
21. `replica_{replica_id}_stage_{stage_id}_busy_time_percent_weighted_mean`: Percentage of time a given replica stage is executing something on device - i.e. not waiting due to scheduling issues or pipeline bubbles.
22. `replica_{replica_id}_stage_{stage_id}_mfu_weighted_mean`: Model FLOPS Utilization (MFU) at a per replica stage level - it tell how much value we are able to extract from the hardware. MFU increases with batch size, reduced bubble time, higher prefill tokens, etc.
23. `request_arrivals_time_series`: Time series of request arrival timestamps.
