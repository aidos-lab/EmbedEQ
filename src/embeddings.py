"Various Embedding methods for dimensionality reduction"

import os

######################################################################
# Silencing UMAP Warnings
import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning


warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
warnings.simplefilter("ignore")

os.environ["KMP_WARNINGS"] = "off"
######################################################################")
from umap import UMAP
from phate import PHATE
from sklearn.manifold import TSNE


def umap(data, hyperparams, seed=0, **kwargs):
    n, d, metric, dim = hyperparams
    operator = UMAP(
        n_neighbors=n,
        min_dist=d,
        n_components=dim,
        metric=metric,
        init="spectral",
        random_state=seed,
    )
    projection = operator.fit_transform(data)
    return projection


def tSNE(data, hyperparams, seed=0, **kwargs):
    perplexity, ee, dim = hyperparams
    operator = TSNE(
        n_components=dim,
        perplexity=perplexity,
        early_exaggeration=ee,
    )
    projection = operator.fit_transform(data)
    return projection


def phate(data, hyperparams, seed=0, **kwargs):
    knn, gamma, dim = hyperparams
    operator = PHATE(
        n_components=dim,
        knn=knn,
        gamma=gamma,
        random_state=seed,
        verbose=0,
    )
    projection = operator.fit_transform(data)
    return projection
