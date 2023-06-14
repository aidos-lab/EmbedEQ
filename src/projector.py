"""Dimensionality reduction methods."""

import argparse
import json
import os
import sys
import warnings
import pickle
import logging

from dotenv import load_dotenv
from umap import UMAP

import embeddings
import data

######################################################################
# Silencing UMAP Warnings
import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from umap import UMAP

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

os.environ["KMP_WARNINGS"] = "off"
######################################################################")


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

    X, C = generator(N=args.num_samples)
    hyperparams = params_json["coordinates"][args.i]
    embedding = getattr(embeddings, args.projector)
    logging.info(f"Using embedding routine {embedding}")
    projection = embedding(X, hyperparams)

    out_file = f"{args.projector}_{args.i}.pkl"
    output_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/projections/"
        + params_json["projector"]
        + "/",
    )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    results = {"projection": projection, "hyperparams": hyperparams}

    out_file = os.path.join(output_dir, out_file)
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    if args.Verbose:

        logging.info(
            "-------------------------------------------------------------------------------------- \n\n"
        )

        logging.info(f"Projection Outfile written to {output_dir}")
        logging.info("\n")
        logging.info(
            "-------------------------------------------------------------------------------------- \n\n"
        )
