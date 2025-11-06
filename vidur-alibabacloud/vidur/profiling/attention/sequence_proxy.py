from typing import List


class SequenceProxy:
    def __init__(self, total_len: int, processed_len: int):
        self._total_len = total_len
        self._processed_len = processed_len
        return

    def get_len(self):
        return self._total_len

    def get_next_prompt_chunk_len(self, chunk_size: int):
        return min(chunk_size, self._total_len - self._processed_len)

    def get_num_prompt_tokens_processed(self):
        return self._processed_len


class SequenceMetadataProxy:
    def __init__(
        self,
        is_prompt: bool,
        total_len: int,
        processed_len: int,
        block_table: List[int],
    ):
        self.is_prompt = is_prompt
        self.prompt_chunk_len = (total_len - processed_len) if is_prompt else None
        self.seq = SequenceProxy(total_len, processed_len)
        self.block_table = block_table
