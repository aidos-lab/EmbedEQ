import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence
from sklearn.metrics import pairwise_distances

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
        "-p",
        "--projector",
        type=str,
        default=params_json["projector"],
        help="Set to the name of projector for dimensionality reduction. ",
    )
    parser.add_argument(
        "-M",
        "--homology_max_dim",
        default=params_json["homology_max_dim"],
        type=int,
        help="Maximum homology dimension for Ripser.py",
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
        default=params_json["random_state"],
        type=int,
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
    # Original Space
    if args.i == -1:
        id_ = "original space"
        generator = getattr(data, args.data)
        X, labels = generator(
            N=args.num_samples,
            random_state=args.seed,
            n_clusters=params_json["num_clusters"],
        )
        if len(X) > 100000:
            print("Data Set is too large to compute pairwise distances")
            sys.exit(-1)

    else:
        in_file = f"{args.projector}_{args.i}.pkl"
        in_dir = os.path.join(
            root,
            "data/"
            + params_json["data_set"]
            + "/projections/"
            + params_json["projector"]
            + "/",
        )
        in_file = os.path.join(in_dir, in_file)
        assert os.path.isfile(in_file), "Invalid Projection"

        with open(in_file, "rb") as f:
            reference = pickle.load(f)
        id_ = reference["hyperparams"]
        X = reference["projection"]

    if params_json["normalize"]:
        if len(X) > 10000:
            print("Data Set is too large to compute pairwise distances")
            sys.exit(-1)
        else:
            D = pairwise_distances(X)

            max_D = D.max()
            X /= max_D

    assert params_json["filtration"] in [
        "rips",
        "alpha",
    ], "Only support Alpha and Vietoris-Rips filtrations currently"

    if params_json["filtration"] == "rips":
        PH = VietorisRipsPersistence
    if params_json["filtration"] == "alpha":
        PH = WeakAlphaPersistence

    # Compute Homology
    dims = tuple(range(args.homology_max_dim + 1))
    model = PH(homology_dimensions=dims)
    X = X.reshape(1, *X.shape)
    dgms = np.squeeze(model.fit_transform(X))
    results = {"diagram": dgms, "hyperparams": id_}

    out_file = f"diagram_{args.i}.pkl"
    out_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/diagrams/"
        + params_json["projector"]
        + "/",
    )

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, out_file)
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )

    logging.info(f"Persistence Diagram {args.i} written to {out_dir}")
    logging.info("\n")
    logging.info(
        "-------------------------------------------------------------------------------------- \n\n"
    )
