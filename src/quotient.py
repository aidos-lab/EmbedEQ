"Cluster Hyperparameters based on the topology of the corresponding embedding."

import argparse
import logging
import os
import pickle
import sys

import numpy as np
from gtda.diagrams import PairwiseDistance
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering

from loaders.factory import load_diagrams, load_parameter_file, project_root_dir
from utils import gtda_pad


def config_filter(cfg):
    return (
        cfg.data.generator == args.data
        and cfg.model.name == args.projector
        and args.num_samples == cfg.data.num_samples
    )


def pairwise_distance(
    diagrams,
    dims,
    metric="bottleneck",
):
    dgms = gtda_pad(diagrams, dims)
    distance_metric = PairwiseDistance(metric=metric, order=1)
    distance_metric.fit(dgms)
    distances = distance_metric.transform(dgms)

    return distances


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

    out_dir = os.path.join(
        root,
        "data/" + args.data + "/" + params.run_name + "/EQC/" + args.projector + "/",
    )

    if args.normalize:
        metric = f"normalized_{args.metric}"
    else:
        metric = args.metric

    # LOAD DIAGRAMS
    keys, diagrams = load_diagrams(condition=config_filter)
    dims = tuple(range(params.topology.homology_max_dim + 1))

    # TOPOLOGICAL DISTANCES #
    distances = pairwise_distance(diagrams, dims=dims, metric=args.metric)
    distance_matrix_ = {"keys": keys, "distances": distances}

    # SAVING DISTANCES
    distances_out_dir = os.path.join(out_dir, "distance_matrices")
    if not os.path.isdir(distances_out_dir):
        os.makedirs(distances_out_dir, exist_ok=True)
    distances_out_file = f"{metric}_pairwise_distances.pkl"
    distances_out_file = os.path.join(distances_out_dir, distances_out_file)
    with open(distances_out_file, "wb") as f:
        pickle.dump(distance_matrix_, f)

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
