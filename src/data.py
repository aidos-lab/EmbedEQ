"""Data set generator methods."""

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def swiss_roll(
    N: int = 1500,
    hole: bool = False,
    connectivity: bool = True,
    n_clusters: int = 6,
    **kwargs,
):
    """Generate Swiss Roll data set."""

    data, color = make_swiss_roll(
        n_samples=N,
        random_state=0,
        hole=hole,
    )

    ## Assign Labels as per https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py
    if connectivity:
        connectivity = kneighbors_graph(data, n_neighbors=10, include_self=False)
    else:
        connectivity = None

    ward = AgglomerativeClustering(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    ).fit(data)
    labels = ward.labels_
    return data, color, labels
