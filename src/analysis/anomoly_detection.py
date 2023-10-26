"Detect Anomalous Embeddings."
import argparse
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from gtda.diagrams import Amplitude, Scaler


def senstivity_filter(cfg):
    """Filtering which models are clustered."""

    return (
        cfg.data.generator == args.data
        and cfg.model.name == args.projector
        and cfg.data.num_samples == args.num_samples
        # and cfg.model.metric == args.projector_metric
        # and cfg.model.min_dist == args.min_dist
    )


def scaler_fn(x):
    return np.max(x)


if __name__ == "__main__":
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src")
    #### SRC IMPORTS ####
    # Load Params
    import utils
    import vis
    from loaders import factory
    from quotient import cluster_models, pairwise_distance

    params = factory.load_parameter_file()
    root = factory.project_root_dir()
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
        "-x",
        "--projector_metric",
        type=str,
        default=params.embedding.hyperparams.metric[0],
        help="Select projector metric.",
    )
    parser.add_argument(
        "-D",
        "--min_dist",
        type=str,
        default=params.embedding.hyperparams.min_dist[0],
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

    args.data, args.projector = utils.format_arguments([args.data, args.projector])

    log_dir = root + f"/experiments/{params.run_name}/logs/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_file = log_dir + f"anomalies_{args.data}_{args.projector}.log"

    logger = logging.getLogger("anomaly_detection_results")
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if args.normalize:
        metric = f"normalized_{args.metric}"
    else:
        metric = args.metric
    # LOAD DIAGRAMS
    keys, diagrams = factory.load_diagrams(condition=senstivity_filter)

    dims = tuple(range(params.topology.homology_max_dim + 1))

    # TOPOLOGICAL DISTANCES #
    distances = pairwise_distance(diagrams, dims=dims, metric=args.metric)
    distance_matrix_ = {"keys": keys, "distances": distances}

    diagrams = utils.gtda_pad(diagrams)
    scaler = Scaler(
        metric=params.topology.diagram_metric,
        function=scaler_fn,
    )
    scaled_diagrams = scaler.fit_transform(diagrams)

    # Landscape Norms
    A = Amplitude(metric=params.topology.diagram_metric, order=2)
    norms = A.fit_transform(scaled_diagrams)
    sigma = np.std(norms)
    mu = np.mean(norms)

    anomalies = []
    embeddings = []
    for i, norm in enumerate(norms):
        if abs(norm - mu) > 2 * sigma:
            anomalies.append(keys[i])
            id = int(re.search(r"\d+$", keys[i]).group())
            X = factory.load_embedding(id)
            embeddings.append(X)

    for i, X in enumerate(embeddings):
        plt.scatter(X.T[0], X.T[1], label=anomalies[i], alpha=0.5)

    plt.legend()
    plt.show()
    logger.info(f"Anomalous Configs: {anomalies}")
    logger.info(f"Anomalous Embeddings: {embeddings}")
