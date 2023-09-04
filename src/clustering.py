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
        "-n",
        "--normalize",
        default=params_json["normalize"],
        type=bool,
        help="Whether to use normalized topological distances.",
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
        "data/"
        + params_json["data_set"]
        + "/diagrams/"
        + params_json["projector"]
        + "/",
    )
    out_dir = os.path.join(
        root,
        "data/" + params_json["data_set"] + "/EQC/" + params_json["projector"] + "/",
    )

    if args.normalize:
        metric = f"normalized_{args.metric}"
    else:
        metric = args.metric

    distances_out_dir = os.path.join(out_dir, "distance_matrices")
    if not os.path.isdir(distances_out_dir):
        os.makedirs(distances_out_dir, exist_ok=True)
    distances_out_file = f"{metric}_pairwise_distances.pkl"
    distances_out_file = os.path.join(distances_out_dir, distances_out_file)

    # If Distances have already been computed
    if not os.path.isfile(distances_out_file):
        # TOPOLOGICAL DISTANCES #
        dims = tuple(range(params_json["homology_max_dim"] + 1))
        keys, distances = pairwise_distance(in_dir, dims=dims, metric=args.metric)
        distance_matrix = {"keys": keys, "distances": distances}
        with open(distances_out_file, "wb") as f:
            pickle.dump(distance_matrix, f)
    else:
        logging.info(f"Distances for {metric} have already been computed.")
        with open(distances_out_file, "rb") as f:
            distance_matrix = pickle.load(f)
        keys = distance_matrix["keys"]
        distances = distance_matrix["distances"]

    # HIERARCHICHAL CLUSTERING #
    model = cluster_models(
        distances,
        linkage=args.linkage,
        distance_threshold=args.dendrogram_cut,
    )
    results = {
        "model": model,
        "hyperparams": {
            "metric": args.metric,
            "cut": args.dendrogram_cut,
            "linkage": args.linkage,
        },
    }
    model_out_dir = os.path.join(out_dir, "models")
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir, exist_ok=True)
    model_out_file = f"embedding_clustering_{metric}_{args.linkage}-linkage_{args.dendrogram_cut}.pkl"
    model_out_file = os.path.join(model_out_dir, model_out_file)

    with open(model_out_file, "wb") as f:
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
