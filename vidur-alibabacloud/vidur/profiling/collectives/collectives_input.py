from random import randint


class CollectivesInput:
    used_comm_ids = set()

    def __init__(
        self,
        num_workers: int,
        num_workers_per_node: int,
        collective_size: int,
        collective: str,
    ):
        self.num_workers = num_workers
        self.num_workers_per_node = num_workers_per_node
        self.collective_size = collective_size
        self.collective = collective
        self.comm_id = self._get_comm_id()

    @classmethod
    def _get_comm_id(cls):
        comm_id = randint(0, 65535)
        while comm_id in cls.used_comm_ids:
            comm_id = randint(0, 65535)
        cls.used_comm_ids.add(comm_id)
        return comm_id

    def is_valid(self, total_gpus_available: int, num_nodes: int):
        if self.num_workers == 1:
            return False

        if self.collective == "send_recv" and self.num_workers != 2:
            return False

        if (
            self.num_workers > self.num_workers_per_node
            and self.num_workers % self.num_workers_per_node != 0
        ):
            return False

        num_nodes_required = self.num_workers // self.num_workers_per_node

        if self.num_workers > total_gpus_available or num_nodes_required > num_nodes:
            return False

        return True
