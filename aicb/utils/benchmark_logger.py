"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import torch
import logging
from utils.timer import Timer
from log_analyzer.log import Log, LogItem


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="LLM_Comm_Benchmark", level=logging.INFO)


class BenchLogger:
    def __init__(self):
        self.comm_log = Log()
        self.enable = True
        self.timer = Timer()
        self.epoch_timer = Timer(use_host_timer=True)
        self.epoch = 0
        self.epoch_timer.start()

    def log_timing(self, name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.timer.start()
                result = func(*args, **kwargs)
                elapsed_time_ms = self.timer.stop()

                log_item = next((item for item in args if isinstance(item, LogItem)))
                if log_item.additional == 'overlap':
                    log_item.elapsed_time = 0
                else:
                    log_item.elapsed_time = elapsed_time_ms
                self.comm_log.add_comm_log(log_item)
                if torch.distributed.get_rank() == 0:
                    logger.info(log_item.view_as_ds_log())
                return result

            return wrapper

        return decorator

    def end_epoch(self, log_item):
        torch.cuda.synchronize()
        elapsed_time_ms = self.epoch_timer.stop()
        if torch.distributed.get_rank() == 0:
            logger.info(
                f"[RANK 0] --------epoch {self.epoch} | micro_step time {elapsed_time_ms:.2f} ---------\n"
            )
        log_item.elapsed_time = elapsed_time_ms
        self.comm_log.add_comm_log(log_item)
        self.epoch += 1
        self.epoch_timer.start()

    def dump_log(self, filename):
        csv_filename = self.comm_log.dump(filename)
        return csv_filename

    def analyze_comm_log(self, print_fn=logger.info):
        return self.comm_log.analyze(print_fn)

    def analyze_comm_time(self, print_fn=logger.info):
        return self.comm_log.analyze_time(print_fn)


bench_logger = BenchLogger()
