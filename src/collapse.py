import argparse
import logging
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

from analysis.collapsers import min_topological_distance
from loaders.factory import (
    load_distances,
    load_model,
    load_parameter_file,
    project_root_dir,
)
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
    params = load_parameter_file()
    root = project_root_dir()

    parser.add_argument(
        "-r",
        "--run_name",
        type=str,
        default=params.run_name,
        help="Identifier for config `yaml`.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=params.data.dataset[0],
        help="Dataset.",
    )
    parser.add_argument(
        "-p",
        "--projector",
        type=str,
        default=params.embedding.model[0],
        help="Choose the type of algorithm to cluster.",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=params.data.num_samples[0],
        help="Choose the type of algorithm to cluster.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default=params.topology.diagram_metric,
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )
    parser.add_argument(
        "-M",
        "--homology_max_dim",
        default=params.topology.homology_max_dim,
        type=int,
        help="Maximum homology dimension for Ripser.py",
    )

    parser.add_argument(
        "-l",
        "--linkage",
        type=str,
        default=params.topology.linkage,
        help="Select linkage algorithm for building Agglomerative Clustering Model.",
    )
    parser.add_argument(
        "-c",
        "--dendrogram_cut",
        type=str,
        default=params.topology.dendrogram_cut,
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )
    parser.add_argument(
        "--normalize",
        default=params.topology.normalize,
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

    keys, distances = load_distances(
        args.data,
        args.projector,
        args.metric,
        args.num_samples,
    )

    model = load_model(
        args.data,
        args.projector,
        args.metric,
        args.num_samples,
        args.linkage,
        args.dendrogram_cut,
    )

    selection = embedding_selector(keys, distances, model)

    out_file = f"selection_{metric}_{args.linkage}-linkage_{args.dendrogram_cut}.pkl"
    out_dir = os.path.join(
        root,
        "data/"
        + args.data
        + "/"
        + params.run_name
        + "/EQC/"
        + args.projector
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
