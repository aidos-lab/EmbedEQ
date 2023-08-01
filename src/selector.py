import argparse
import json
import os
import pickle
import sys
import logging

import numpy as np
from dotenv import load_dotenv


def embedding_selector(keys, distances, model):

    original = [True if type(key) is str else False for key in keys]
    cluster_labels = list(model.labels_)
    N = len(cluster_labels)
    O = np.array(range(N))[original][0]
    selection = {}
    for label in np.unique(cluster_labels):
        mask = np.where(cluster_labels == label, True, False)
        items = np.array(range(N))[mask]
        # Remove Cluster containing original
        if O in items:
            continue
        # Min Distance from original space
        min_dist = np.min(distances[O][items])
        i = np.where(distances[O] == min_dist)[0][0]
        selection[label] = keys[i]
    return selection


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src/")

    JSON_PATH = os.getenv("params")
    assert os.path.isfile(JSON_PATH), "Please configure .env to point to params.json"
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)

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

    if args.normalize:
        metric = f"normalized_{args.metric}"
    else:
        metric = args.metric

    projections_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/projections/"
        + params_json["projector"]
        + "/",
    )
    diagrams_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/diagrams/"
        + params_json["projector"]
        + "/",
    )
    distances_in_file = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/EQC/"
        + params_json["projector"]
        + "/distance_matrices/"
        + f"{metric}_pairwise_distances.pkl",
    )
    model_in_file = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/EQC/"
        + params_json["projector"]
        + "/models/"
        + f"embedding_clustering_{metric}_{args.linkage}-linkage_{args.dendrogram_cut}.pkl",
    )

    with open(distances_in_file, "rb") as D:
        reference = pickle.load(D)

    keys, distances = reference["keys"], reference["distances"]

    with open(model_in_file, "rb") as M:
        model = pickle.load(M)["model"]

    selection = embedding_selector(keys, distances, model)

    out_file = f"selection_{metric}_{args.linkage}-linkage_{args.dendrogram_cut}.pkl"
    out_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/EQC/"
        + params_json["projector"]
        + "/selected_embeddings/",
    )

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_file)

    with open(out_file, "wb") as f:
        pickle.dump(selection, f)

    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )

    logging.info(
        f"Selected Models for {metric} and {args.linkage}-linkage written to{out_dir}."
    )
    logging.info("\n")
    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )
