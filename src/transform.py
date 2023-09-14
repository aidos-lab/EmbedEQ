"Homology Transform for Embeddings"

import argparse
import logging
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence
from omegaconf import OmegaConf
from sklearn.metrics import pairwise_distances

import data
from loaders.factory import (
    LoadClass,
    load_config,
    load_parameter_file,
    project_root_dir,
)
from utils import format_arguments

if __name__ == "__main__":
    # Load Params
    params = load_parameter_file()
    root = project_root_dir()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--run_name",
        type=str,
        default=params.run_name,
        help="Identifier for config `yaml`.",
    )
    parser.add_argument(
        "-i",
        type=int,
        default=0,
        help="Identifier for config `yaml`.",
    )
    parser.add_argument(
        "-M",
        "--homology_max_dim",
        default=params.topology.homology_max_dim,
        type=int,
        help="Maximum homology dimension for Ripser.py",
    )

    parser.add_argument(
        "-s",
        "--seed",
        default=params.data.seed,
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

    folder = project_root_dir() + f"/experiments/{args.run_name}/configs/"
    cfg = load_config(args.i, folder)

    in_file = f"embedding_{args.i}.pkl"
    in_dir = os.path.join(
        root,
        "data/"
        + cfg.data.generator
        + "/"
        + args.run_name
        + "/projections/"
        + cfg.model.name
        + "/",
    )
    in_file = os.path.join(in_dir, in_file)
    assert os.path.isfile(in_file), "Invalid Projection"

    with open(in_file, "rb") as f:
        D = pickle.load(f)
        X = D["projection"]

    if params.topology.normalize:
        if len(X) > 10005:
            print("Data Set is too large to compute pairwise distances")
            sys.exit(-1)
        else:
            D = pairwise_distances(X)
            max_D = D.max()
            X /= max_D

    assert params.topology.filtration in [
        "rips",
        "alpha",
    ], "Only support Alpha and Vietoris-Rips filtrations currently"

    if params.topology.filtration == "rips":
        PH = VietorisRipsPersistence
    if params.topology.filtration == "alpha":
        PH = WeakAlphaPersistence

    # Compute Homology
    dims = tuple(range(args.homology_max_dim + 1))
    model = PH(homology_dimensions=dims)
    X = X.reshape(1, *X.shape)
    dgms = np.squeeze(model.fit_transform(X))
    results = {"diagram": dgms}

    out_file = f"diagram_{args.i}.pkl"
    out_dir = os.path.join(
        root,
        "data/"
        + cfg.data.generator
        + "/"
        + args.run_name
        + "/diagrams/"
        + cfg.model.name
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
