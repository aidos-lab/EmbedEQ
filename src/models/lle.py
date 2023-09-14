from dataclasses import dataclass

from sklearn.manifold import LocallyLinearEmbedding as LLE

from .base import BaseConfig, BaseProjector


@dataclass
class LLEConfig(BaseConfig):
    n_neighbors: int = 15
    reg: float = 0
    metric: str = "euclidean"
    dim: int = 2


class LLEProjector(BaseProjector):
    def __init__(self, config: LLEConfig):
        super().__init__(config)

    def project(self, data):
        operator = LLE(
            n_neighbors=self.config.n_neighbors,
            reg=self.config.reg,
            n_components=self.config.dim,
        )
        return operator.fit_transform(data)
