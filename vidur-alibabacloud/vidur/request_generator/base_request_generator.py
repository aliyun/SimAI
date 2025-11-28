import json
from abc import ABC, abstractmethod
from typing import List

from vidur.config import BaseRequestGeneratorConfig
from vidur.entities import Request


class BaseRequestGenerator(ABC):

    def __init__(self, config: BaseRequestGeneratorConfig):
        self.config = config

    @abstractmethod
    def generate_requests(self) -> List[Request]:
        pass

    def generate(self) -> List[Request]:
        requests = self.generate_requests()
        return requests
