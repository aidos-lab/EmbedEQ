from dataclasses import dataclass

from sklearn.manifold import TSNE

from .base import BaseConfig, BaseProjector


@dataclass
class TSNEConfig(BaseConfig):
    n_neighbors: int = 15
    early_exaggeration: int = 10
    dim: int = 2
    metric: str = "euclidean"


class TSNEProjector(BaseProjector):
    def __init__(self, config: TSNEConfig):
        super().__init__(config)

    def project(self, data):
        operator = TSNE(
            perplexity=self.config.n_neighbors,
            early_exaggeration=self.config.early_exaggeration,
            n_components=self.config.dim,
            metric=self.config.metric,
        )

        return operator.fit_transform(data)
