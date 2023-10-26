from dataclasses import dataclass

from umap import UMAP

from .base import BaseConfig, BaseProjector


@dataclass
class UMAPConfig(BaseConfig):
    n_neighbors: int = 15
    min_dist: float = 0.1
    init: str = "spectral"
    metric: str = "euclidean"
    dim: int = 2


class UMAPProjector(BaseProjector):
    def __init__(self, config: UMAPConfig):
        super().__init__(config)

    def project(self, data):
        operator = UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            n_components=self.config.dim,
            metric=self.config.metric,
            init=self.config.init,
            random_state=self.config.seed,
        )

        return operator.fit_transform(data)
