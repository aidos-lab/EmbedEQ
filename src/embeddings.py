"Various Embedding methods for dimensionality reduction"

import os
import scanpy as sc

######################################################################
# Silencing UMAP Warnings
import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning


warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

os.environ["KMP_WARNINGS"] = "off"
######################################################################")
from umap import UMAP


def umap(data, hyperparams, seed=0, **kwargs):
    n, d, dim, metric = hyperparams
    umap = UMAP(
        n_neighbors=n,
        min_dist=d,
        n_components=dim,
        metric=metric,
        init="spectral",
        random_state=seed,
    )
    projection = umap.fit_transform(data)
    return projection
