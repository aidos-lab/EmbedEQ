"Utility functions."

import itertools
import json
import os
import pickle
from dotenv import load_dotenv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def parameter_coordinates(hyper_params: dict, embedding):

    # TODO: Generalize formatting parameter grid search for different Embeddings
    assert embedding in ["umap", "tSNE", "phate"], f"{embedding} is not yet supported."
    dim = hyper_params["dim"]
    if embedding == "umap":
        N = hyper_params["n_neighbors"]
        d = hyper_params["min_dist"]
        m = hyper_params["metric"]
        reported_params = {
            "n_neighbors": N,
            "min_dist": d,
            "metric": m,
            "dim": dim,
        }
        coordinates = list(itertools.product(N, d, m, dim))

    if embedding == "tSNE":
        N = hyper_params["n_neighbors"]
        ee = hyper_params["early_exaggeration"]

        reported_params = {
            "perplexity": N,
            "early_exaggeration": ee,
            "dim": dim,
        }
        coordinates = list(itertools.product(N, ee, dim))

    if embedding == "phate":
        knn = hyper_params["n_neighbors"]
        gamma = hyper_params["gamma"]

        reported_params = {
            "knn": knn,
            "gamma": gamma,
            "dim": dim,
        }
        coordinates = list(itertools.product(knn, gamma, dim))

    return coordinates, reported_params


def assign_labels(data, n_clusters, k=10):
    """Assign Labels for Sklearn Data Sets based on KNN Graphs"""

    connectivity = kneighbors_graph(data, n_neighbors=k, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    ).fit(data)
    labels = ward.labels_
    return labels


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


def subplot_grid(dir):
    hashmap = {}
    neighbors, dists = [], []
    coords = []
    for umap in os.listdir(dir):
        with open(f"{dir}/{umap}", "rb") as f:
            D = pickle.load(f)
        projection = D["projection"]
        params = D["hyperparams"][:2]
        coords.append(params)
        hashmap[str(params[:2]).replace(" ", "")] = projection
        if D["hyperparams"][0] not in neighbors:
            neighbors.append(D["hyperparams"][0])
        if D["hyperparams"][1] not in dists:
            dists.append(D["hyperparams"][1])

    neighbors.sort()
    dists.sort()
    coords.sort()
    return hashmap, neighbors, dists, coords


def generate_hex_variations(n, hex_color="#636EFA"):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(*rgb_color)

    original_rgb = hex_to_rgb(hex_color)
    variations = []

    for i in range(1, n + 1):
        r, g, b = original_rgb
        # You can modify the values below to create different color variations.
        r += 2 * i
        g += 10 * i
        new_rgb = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        variations.append(rgb_to_hex(new_rgb))
    assert len(variations) == len(np.unique(variations))
    return variations


def embedding_coloring(color_map, hex_color="#636EFA"):
    count = Counter(color_map.values())[hex_color]

    new_colors = generate_hex_variations(
        count,
        hex_color,
    )

    C = 0
    for key in color_map:
        if color_map[key] == hex_color:
            color_map[key] = new_colors[C]
            C += 1

    return color_map
