"""Driver file for generating embeddings."""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

import data
import models
from config import configs, projectors
from loaders.factory import (
    LoadClass,
    load_config,
    load_parameter_file,
    project_root_dir,
)

if __name__ == "__main__":
    # Load Params
    params = load_parameter_file()
    parser = argparse.ArgumentParser()
    root = project_root_dir()
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
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )
    args = parser.parse_args()
    this = sys.modules[__name__]

    # Read in correct config
    folder = project_root_dir() + f"/experiments/{args.run_name}/configs/"
    cfg = load_config(args.i, folder)

    # Get Config Class and Projector class for given model
    C = getattr(models, configs[cfg.model.name])
    M = getattr(models, projectors[cfg.model.name])

    model_cfg = LoadClass.instantiate(C, cfg.model)
    model = M(model_cfg)

    # Load/Generate Data
    generator = getattr(data, cfg.data.generator)
    logging.info(f"Using generator routine {generator}")
    X, labels = generator(
        N=cfg.data.num_samples,
        n_clusters=params.data.num_clusters,
        random_state=cfg.data.seed,
    )

    projection = model.project(X)

    out_file = f"embedding_{args.i}.pkl"
    out_dir = os.path.join(
        root,
        "data/"
        + cfg.data.generator
        + "/"
        + args.run_name
        + "/projections/"
        + cfg.model.name
        + "/",
    )

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    results = {"projection": projection}

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
