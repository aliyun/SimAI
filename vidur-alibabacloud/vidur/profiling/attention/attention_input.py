class AttentionInput:
    def __init__(
        self,
        prefill_chunk_size: int,
        kv_cache_size: int,
        batch_size: int,
        is_prefill: bool,
    ):
        self.prefill_chunk_size = prefill_chunk_size
        self.kv_cache_size = kv_cache_size
        self.batch_size = batch_size
        self.is_prefill = is_prefill

    def is_valid(self, max_seq_len: int):
        if self.is_prefill:
            if self.batch_size != 1:
                return False
            elif self.prefill_chunk_size == 0:
                return False
            elif self.prefill_chunk_size + self.kv_cache_size > max_seq_len:
                return False
        else:
            if self.prefill_chunk_size > 0:
                return False
            elif self.kv_cache_size == 0:
                return False
            elif self.kv_cache_size > max_seq_len:
                return False
        return True

    def is_under_memory_limit(self, max_num_tokens: int):
        return (
            self.batch_size * (self.kv_cache_size + self.prefill_chunk_size)
            <= max_num_tokens
        )

    def __str__(self):
        return f"prefill_chunk_size: {self.prefill_chunk_size}, kv_cache_size: {self.kv_cache_size}, batch_size: {self.batch_size}, is_prefill: {self.is_prefill}"
