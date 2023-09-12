from dataclasses import dataclass

from sklearn.manifold import Isomap

from .base import BaseConfig, BaseProjector


@dataclass
class IsomapConfig(BaseConfig):
    n_neighbors: int = 15
    metric: str = "euclidean"
    dim: int = 2


class IsomapProjector(BaseProjector):
    def __init__(self, config: IsomapConfig):
        super().__init__(config)

    def project(self, data):
        operator = Isomap(
            n_neighbors=self.config.n_neighbors,
            n_components=self.config.dim,
            metric=self.config.metric,
        )
        return operator.fit_transform(data)
