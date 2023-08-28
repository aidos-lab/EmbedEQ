"Functions to determine token embedding selection from an equivalency class"

import json
import os

import numpy as np
from dotenv import load_dotenv


def min_topological_distance(keys, labels, distances, id_, **kwargs):
    # TODO: Select based on id_
    assert len(distances) > 0
    original_space_idx = [True if type(key) is str else False for key in keys]
    original_distances = distances[original_space_idx][0]
    selection = {}
    for label in np.unique(labels):
        mask = np.where(labels == label, True, False)
        class_distances = original_distances[mask]

        # If EQ class is > 1 and contains original space
        if mask[original_space_idx]:
            if len(class_distances) == 1:
                continue
            else:
                # Force selecting another embedding
                class_distances[class_distances == 0] = np.infty
        min_dist = np.min(class_distances)
        idx = np.where(original_distances == min_dist)[0][0]
        selection[label] = keys[idx]
    return selection


def best_clustering_score(
    keys,
    labels,
    **kwargs,
):
    # TODO: Implement best clustering accuracy
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score

    import data
    from utils import subplot_grid

    load_dotenv()
    root = os.getenv("root")
    JSON_PATH = os.getenv("params")
    assert os.path.isfile(JSON_PATH), "Please configure .env to point to params.json"
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)

    generator = getattr(data, params_json["data_set"])
    _, labels = generator()
    in_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/projections/"
        + params_json["projector"]
        + "/",
    )
    n_clusters = len(np.unique(labels))
    hashmap, _, _, coords = subplot_grid(in_dir)

    scores = {}
    for coord in coords:
        ref = str(coord).replace(" ", "")
        # Get Embedding
        X = hashmap[ref]
        # Fit Kmeans
        pred = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(X)
        score = accuracy_score(labels, pred)

        scores[ref] = score

    return scores
