"Utility functions."

import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def parameter_coordinates(hyper_params: dict, embedding, sample_size):

    # TODO: Generalize formatting parameter grid search for different Embeddings
    assert embedding in [
        "umap",
        "tSNE",
        "phate",
        "isomap",
        "LLE",
    ], f"{embedding} is not yet supported."
    dim = hyper_params["dim"]

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
        metric = hyper_params["metric"]

        reported_params = {
            "knn": knn,
            "gamma": gamma,
            "metric": metric,
            "dim": dim,
        }
        coordinates = list(itertools.product(knn, gamma, metric, dim))

    if embedding == "isomap":
        N = hyper_params["n_neighbors"]
        m = hyper_params["metric"]

        reported_params = {
            "n_neighbors": N,
            "metric": m,
            "dim": dim,
        }
        coordinates = list(itertools.product(N, m, dim))

    if embedding == "LLE":
        N = hyper_params["n_neighbors"]
        reg = hyper_params["reg"]

        reported_params = {
            "n_neighbors": N,
            "reg": reg,
            "dim": dim,
        }
        coordinates = list(itertools.product(N, reg, dim))

    return coordinates, reported_params




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


# TODO: Rename and document


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
