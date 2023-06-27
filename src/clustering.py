"Cluster Hyperparameters based on the topology of the corresponding embedding."


import argparse
import json
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering

from topology import pairwise_distance


def cluster_models(
    distances,
    linkage="average",
    distance_threshold=0.5,
):

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage=linkage,
        compute_distances=True,
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    model.fit(distances)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    load_dotenv()
    root = os.getenv("root")
    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=params_json["data_set"],
        help="Specify the data set.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default=params_json["diagram_metric"],
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )

    parser.add_argument(
        "-l",
        "--linkage",
        type=str,
        default=params_json["linkage"],
        help="Select linkage algorithm for building Agglomerative Clustering Model.",
    )

    parser.add_argument(
        "-c",
        "--dendrogram_cut",
        type=str,
        default=params_json["dendrogram_cut"],
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )

    parser.add_argument(
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    in_dir = os.path.join(
        root,
        "data/" + params_json["data_set"] + "/diagrams/",
    )

    keys, distances = pairwise_distance(in_dir, metric=args.metric)

    labels = []
    coords = []
    for i, key in enumerate(keys):
        if type(key) == str:
            labels.append(key)
            original = i
            idx = list(distances[i])
            idx.pop(i)
        else:
            labels.append(key[:2])
            coords.append(key[:2])
    coords = np.array(coords)

    model = cluster_models(
        distances,
        linkage=args.linkage,
        distance_threshold=args.dendrogram_cut,
    )

    out_file = f"embedding_clustering_{args.metric}_{args.dendrogram_cut}.pkl"
    out_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/clusterings/"
        + params_json["projector"]
        + "/",
    )

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    results = {
        "model": model,
        "hyperparams": {"metric": args.metric, "cut": args.dendrogram_cut},
    }

    out_file = os.path.join(out_dir, out_file)
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )

    logging.info(
        f"Embedding clustering generated using {args.metric} distances between persistence diagrams."
    )
    logging.info("\n")
    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )
