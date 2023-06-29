"""Data set generator methods."""

import json
import os
import pickle

import numpy as np
import scanpy as sc
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_swiss_roll
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


def scanpy():
    sc.settings.verbosity = 0
    root = os.getenv("root")
    JSON_PATH = os.getenv("params")
    assert os.path.isfile(JSON_PATH), "Please configure .env to point to params.json"
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)

    file = os.path.join(root, params_json["path_to_raw_data"])
    assert os.path.isfile(file), "Local Data Copy does not exist"

    adata = sc.read_h5ad(file)
    adata.X = adata.layers["log1p_norm"]
    adata.var["highly_variable"] = adata.var["highly_deviant"]

    sc.pp.pca(
        adata,
        svd_solver="arpack",
        n_comps=params_json["num_pca_components"],
        use_highly_variable=True,
    )
    pca = adata.obsm["X_pca"]
    labels = adata.obs[params_json["scanpy_color_label"]]
    results = {"pca": pca, "labels": labels}

    out_dir = os.path.join(root, f"data/local_copies/pca/")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    name = params_json["data_set"]
    pca_out_file = os.path.join(out_dir, f"{name}_pca.pkl")

    with open(pca_out_file, "wb") as f:
        pickle.dump(results, f)
    return pca_out_file
    # Other option is to just return PCA and use standard pipeline
