"Cluster Hyperparameters based on the topology of the corresponding embedding."


import argparse
import json
import os
import sys
from sklearn.cluster import AgglomerativeClustering

from dotenv import load_dotenv
from utils import pairwise_distance, plot_dendrogram


def cluster_models(
    distances,
    metric,
    p=3,
    distance_threshold=0.5,
    plot=True,
):

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        compute_distances=True,
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    model.fit(distances)

    # What percentage of Labels do you want to visualize?
    if plot:
        plot_dendrogram(
            model=model,
            labels=None,
            distance=metric,
            truncate_mode="level",
            p=p,
            distance_threshold=distance_threshold,
        )

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    load_dotenv()
    root = os.getenv("root")
    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=params_json["data_set"],
        help="Specify the data set.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default=params_json["diagram_metric"],
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )

    parser.add_argument(
        "-c",
        "--dendrogram_cut",
        type=str,
        default=params_json["dendrogram_cut"],
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
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

    in_dir = os.path.join(
        root,
        "data/" + params_json["data_set"] + "/diagrams/",
    )

    keys, distances = pairwise_distance(in_dir, metric=args.metric)

    model = cluster_models(
        distances, args.metric, distance_threshold=args.dendrogram_cut
    )
