"""Dimensionality reduction methods."""

import argparse
import json
import os
import sys
import numpy as np

import pickle
import logging

from dotenv import load_dotenv

import embeddings
import data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    load_dotenv()
    root = os.getenv("root")
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
        "-n",
        "--num_samples",
        default=params_json["num_samples"],
        type=int,
        help="Set number of samples in data set",
    )
    parser.add_argument(
        "-c",
        "--num_clusters",
        default=params_json["num_clusters"],
        type=int,
        help="Set number of samples in data set",
    )
    parser.add_argument(
        "-p",
        "--projector",
        type=str,
        default=params_json["projector"],
        help="Set to the name of projector for dimensionality reduction. ",
    )
    parser.add_argument(
        "-i",
        type=int,
        default=0,
        help="Position in coordinate space.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=params_json["random_state"],
        help="Random Seed for reproducing Sklearn Datasets.",
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

    generator = getattr(data, args.data)
    logging.info(f"Using generator routine {generator}")
    X, labels = generator(
        N=args.num_samples,
        n_clusters=args.num_clusters,
        random_state=args.seed,
    )
    # If classes are automatically generated, reset params file
    params_json["num_clusters"] = len(np.unique(labels))
    params_json["num_samples"] = len(X)
    with open(JSON_PATH, "w") as f:
        json.dump(params_json, f, indent=4)

    hyperparams = params_json["coordinates"][args.i]

    embedding = getattr(embeddings, args.projector)
    logging.info(f"Using embedding routine {embedding}")
    projection = embedding(X, hyperparams)

    out_file = f"{args.projector}_{args.i}.pkl"
    out_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/projections/"
        + params_json["projector"]
        + "/",
    )

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    results = {"projection": projection, "hyperparams": hyperparams}

    out_file = os.path.join(out_dir, out_file)
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )

    logging.info(f"Projection {args.i} written to {out_dir}")
    logging.info("\n")
    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )
