"Utility functions."

import itertools
import json
import os
import pickle
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def parameter_coordinates(hyper_params: dict, embedding):

    assert embedding in ["umap", "tSNE"], f"{embedding} is not yet supported."
    if embedding == "umap":
        N = hyper_params["n_neighbors"]
        d = hyper_params["min_dist"]
        n = hyper_params["dim"]
        m = hyper_params["metric"]

        coordinates = list(itertools.product(N, d, n, m))

    return coordinates


def get_diagrams(dir):
    assert os.path.isdir(
        dir
    ), "Please first compute diagrams by running `make homology`"

    diagrams = []
    keys = []
    for file in os.listdir(dir):
        if file.endswith(".pkl"):
            file = os.path.join(dir, file)
            with open(file, "rb") as f:
                reference = pickle.load(f)
            diagram = reference["diagram"]
            diagrams.append(diagram)
            hyperparams = reference["hyperparams"]
            keys.append(hyperparams)

    return keys, diagrams


def convert_to_gtda(diagrams):
    homology_dimensions = (0, 1)

    slices = {
        dim: slice(None) if (dim) else slice(None, -1) for dim in homology_dimensions
    }
    Xt = [
        {dim: diagram[dim][slices[dim]] for dim in homology_dimensions}
        for diagram in diagrams
    ]
    start_idx_per_dim = np.cumsum(
        [0]
        + [
            np.max([len(diagram[dim]) for diagram in Xt] + [1])
            for dim in homology_dimensions
        ]
    )
    min_values = [
        min(
            [
                np.min(diagram[dim][:, 0]) if diagram[dim].size else np.inf
                for diagram in Xt
            ]
        )
        for dim in homology_dimensions
    ]
    min_values = [min_value if min_value != np.inf else 0 for min_value in min_values]
    n_features = start_idx_per_dim[-1]
    Xt_padded = np.empty((len(Xt), n_features, 3), dtype=float)

    for i, dim in enumerate(homology_dimensions):
        start_idx, end_idx = start_idx_per_dim[i : i + 2]
        padding_value = min_values[i]
        # Add dimension as the third elements of each (b, d) tuple globally
        Xt_padded[:, start_idx:end_idx, 2] = dim
        for j, diagram in enumerate(Xt):
            subdiagram = diagram[dim]
            end_idx_nontrivial = start_idx + len(subdiagram)
            # Populate nontrivial part of the subdiagram
            if len(subdiagram) > 0:
                Xt_padded[j, start_idx:end_idx_nontrivial, :2] = subdiagram
            # Insert padding triples
            Xt_padded[j, end_idx_nontrivial:end_idx, :2] = [padding_value] * 2

    return Xt_padded


def plot_dendrogram(model, labels, distance, p, distance_threshold, **kwargs):
    """Create linkage matrix and then plot the dendrogram for Hierarchical clustering."""
    load_dotenv()
    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    d = dendrogram(
        linkage_matrix,
        p=p,
        distance_sort=True,
        labels=labels,
        color_threshold=distance_threshold,
    )
    if params_json["normalize"]:
        distance = "normalized " + distance
    for leaf, leaf_color in zip(plt.gca().get_xticklabels(), d["leaves_color_list"]):
        leaf.set_color(leaf_color)
    plt.title(f"Persistence Diagrams Clustering")
    plt.xlabel("Embeddings")
    plt.ylabel(f"{distance} distance")
    plt.show()
    return d
