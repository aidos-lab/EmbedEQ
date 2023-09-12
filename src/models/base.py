from abc import ABC, abstractmethod
from dataclasses import dataclass, fields


@dataclass
class BaseConfig:
    name: str = "base"
    seed: int = 150


class BaseProjector(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def project(self):
        """
        This method should be implemented by the user.
        Specify how to unpack parameters from a customized
        config class and pass to a known projection method.
        """
        raise NotImplementedError()
