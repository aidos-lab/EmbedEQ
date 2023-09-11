import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv

from selection_criteria import *
from utils import format_arguments


def embedding_selector(
    keys,
    distances,
    model,
    selection_fn=min_topological_distance,
    id_="original space",
):

    assert len(keys) == len(
        model.labels_
    ), "Mismatch between hyperparamter keys and model labels"

    selection = selection_fn(
        keys=keys,
        labels=model.labels_,
        distances=distances,
        id_=id_,
    )

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
        "-p",
        "--projector",
        type=str,
        default=params_json["projector"],
        help="Set to the name of projector for dimensionality reduction. ",
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

    data_, projector_ = format_arguments([args.data, args.projector])
    if args.normalize:
        metric = f"normalized_{args.metric}"
    else:
        metric = args.metric

    projections_dir = os.path.join(
        root,
        "data/"
        + data_
        + "/"
        + params_json["run_name"]
        + "/projections/"
        + projector_
        + "/",
    )
    diagrams_dir = os.path.join(
        root,
        "data/"
        + data_
        + "/"
        + params_json["run_name"]
        + "/diagrams/"
        + projector_
        + "/",
    )
    distances_in_file = os.path.join(
        root,
        "data/"
        + data_
        + "/"
        + params_json["run_name"]
        + "/EQC/"
        + projector_
        + "/distance_matrices/"
        + f"{metric}_pairwise_distances.pkl",
    )
    model_in_file = os.path.join(
        root,
        "data/"
        + data_
        + "/"
        + params_json["run_name"]
        + "/EQC/"
        + projector_
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
        + data_
        + "/"
        + params_json["run_name"]
        + "/EQC/"
        + projector_
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
