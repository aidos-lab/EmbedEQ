from dataclasses import dataclass

from phate import PHATE

from .base import BaseConfig, BaseProjector


@dataclass
class PHATEConfig(BaseConfig):
    n_neighbors: int = 15
    gamma: float = 0
    metric: str = "euclidean"
    dim: int = 2


class PHATEProjector(BaseProjector):
    def __init__(self, config: PHATEConfig):
        super().__init__(config)

    def project(self, data):
        operator = PHATE(
            knn=self.config.n_neighbors,
            gamma=self.config.gamma,
            knn_dist=self.config.metric,
            n_components=self.config.dim,
            random_state=self.config.seed,
            verbose=0,
        )
        return operator.fit_transform(data)
