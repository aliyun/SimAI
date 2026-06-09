from io import StringIO
import dataclasses
from typing import List

    
@dataclasses.dataclass
class WorkItem:
    name: str = dataclasses.field(default="none")
    placeholder: int = dataclasses.field(default=-1)
    forward_compute_time: int = dataclasses.field(default=1)
    forward_comm: str = dataclasses.field(default="NONE")
    forward_comm_size: int = dataclasses.field(default=0)
    backward_compute_time: int = dataclasses.field(default=0)
    backward_comm: str = dataclasses.field(default="NONE")
    backward_comm_size: int = dataclasses.field(default=0)
    dp_compute_time: int = dataclasses.field(default=0)
    dp_comm: str = dataclasses.field(default="NONE")
    dp_comm_size: int = dataclasses.field(default=0)
    process_time: int = dataclasses.field(default=100)

    def to_string(self):
        return f"{self.name} {self.placeholder} {self.forward_compute_time} {self.forward_comm} {self.forward_comm_size} {self.backward_compute_time} {self.backward_comm} {self.backward_comm_size} {self.dp_compute_time} {self.dp_comm} {self.dp_comm_size} {self.process_time}"
    
class SimAIWorkload:
    def __init__(self, tp_size: int, ep_size: int, pp_size: int, vpp_size: int, ga_num: int, world_size: int, pp_comm: int):
        self.training_loop_parallelization_type: str = 'HYBRID_TRANSFORMER_FWD_IN_BCKWD'
        self.model_parallel_NPU_group: int = tp_size
        self.ep_size: int = ep_size
        self.pp_size: int = pp_size
        # vpp_size为num_layers
        # vpp_size is num_layers
        self.vpp_size: int = vpp_size
        self.ga: int = ga_num
        self.all_gpus: int = world_size
        self.checkpoints: int = 0
        self.checkpoint_initiates: int = 0
        self.pp_comm = pp_comm
        self.work_items: List[WorkItem] = []

    def append_work_item(self, item: WorkItem):
        self.work_items.append(item)

    def flush(self):
        self.work_items.clear()

    def dump_str(self):
        buffer: StringIO = StringIO()
        firstline = (f"{self.training_loop_parallelization_type} "
            f"model_parallel_NPU_group: {self.model_parallel_NPU_group} "
            f"ep: {self.ep_size} "
            f"pp: {self.pp_size} "
            f"vpp: {self.vpp_size} "
            f"ga: {self.ga} all_gpus: {self.all_gpus} "
            f"checkpoints: {self.checkpoints} "
            f"checkpoint_initiates: {self.checkpoint_initiates} "
            f"pp_comm: {self.pp_comm}"
        ) + "\n"
        buffer.write(firstline)
        buffer.write(f"{len(self.work_items)}\n")
        for item in self.work_items:
            buffer.write(item.to_string() + "\n")
        return buffer.getvalue()

    def dump_file(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.dump_str())
