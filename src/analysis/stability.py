"Tracking stability of clusters as you vary sample size."

import argparse
import itertools
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering

if __name__ == "__main__":

    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "/src/")
    from loaders.factory import load_distances, load_parameter_file
    from utils import format_arguments

    params = load_parameter_file()
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

    args.data, args.projector = format_arguments([args.data, args.projector])

    # LOGGING
    log_dir = root + f"/experiments/{params.run_name}/logs/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_file = log_dir + f"stability_{args.data}_{args.projector}.log"

    logger = logging.getLogger("stability_results")
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if args.normalize:
        args.metric = f"normalized_{args.metric}"
    matrices = {}
    sizes = params.data.num_samples
    for sample_size in sizes:
        key, distances = load_distances(
            args.data, args.projector, args.metric, sample_size
        )
        matrices[sample_size] = distances

    pairs = list(itertools.combinations(matrices, 2))

    for x, y in pairs:
        logger.info(f"({x,y})")
        row_correlations = []
        X = matrices[x]
        Y = matrices[y]
        for i in range(len(X)):
            result = np.corrcoef(
                X[i],
                Y[i],
            )
            val = np.min(result)
            row_correlations.append(val)
        logger.info(f"Average over all rows: {np.mean(row_correlations)}")
        logger.info(f"Median Correlation: {np.median(row_correlations)}")
        logger.info(f"Max Correlation: {np.max(row_correlations)}")
        logger.info(f"Min Correlation: {np.min(row_correlations)} ")
        logger.info("\n")

    # for k in range(2, 5):
    #     print(f"K = {k}")
    #     model = AgglomerativeClustering(
    #         metric="precomputed", n_clusters=k, linkage="single"
    #     )

    # # Pearson correlation coefficient computed over the pairs of pairwise distances is reported

    # model = HDBSCAN(
    #     metric="precomputed",
    #     min_cluster_size=2,
    #     min_samples=1,
    #     cluster_selection_epsilon=0.8,
    # )

    # Pairwise Clustering Comparison
    # for i in range(len(matrices) - 1):
    #     # Model 1
    #     distances1 = matrices[i]
    #     sample_size1 = sizes[i]
    #     model.fit(distances1)
    #     labels1 = model.labels_
    #     print(labels1)

    #     # Model 2
    #     distances2 = matrices[i + 1]
    #     sample_size2 = sizes[i + 1]
    #     model.fit(distances2)
    #     labels2 = model.labels_
    #     print(labels2)

    #     ar_score = adjusted_rand_score(labels1, labels2)
    #     mi_score = adjusted_mutual_info_score(labels1, labels2)

    #     print(f"Sample Size Comparison: {(sample_size1,sample_size2)}")
    #     print(f"Adjusted RAND Score: {ar_score}")
    #     print(f"Adjusted Mutual Info Score: {mi_score}")
    #     print()
