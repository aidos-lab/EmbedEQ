import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def get_cluster_color(hyperparams, keys):
    idx = np.argwhere(keys == hyperparams)
    return idx


def make_subplot_specs(num_rows, num_cols):

    skeleton = np.full(shape=(num_rows + 1, num_cols), fill_value={})
    formatted = [x.tolist() for x in skeleton]
    formatted[-1] = [{"colspan": num_cols} if i == 0 else None for i in range(num_cols)]
    return formatted


def visualize_clustered_umaps(dir, model, keys, distances):
    """
    Create a grid visualization of UMAP projections according .

    Parameters:
    -----------
    dir : str
        The directory containing the UMAP projections.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Plotly figure object representing the UMAP grid visualization.
    """

    neighbors, dists = [], []
    for umap in os.listdir(dir):
        with open(f"{dir}/{umap}", "rb") as f:
            params = pickle.load(f)
        if params["hyperparams"][0] not in neighbors:
            neighbors.append(params["hyperparams"][0])
        if params["hyperparams"][1] not in dists:
            dists.append(params["hyperparams"][1])
        neighbors.sort()
        dists.sort()

    spec = make_subplot_specs(len(dists), len(neighbors))
    fig = make_subplots(
        rows=len(dists) + 1,
        cols=len(neighbors),
        column_titles=list(map(str, neighbors)),
        x_title="n_neighbors",
        row_titles=list(map(str, dists)) + ["Dendrogram"],
        y_title="min_dist",
        specs=spec,
    )
    labels = []
    coords = []
    for i, key in enumerate(keys):
        if type(key) == str:
            labels.append(key)
            # original = i
            idx = list(distances[i])
            idx.pop(i)
        else:
            labels.append(key[:2])
            coords.append(key[:2])
    coords = np.array(coords)

    plotly_labels = [f"[{i[0]},{i[1]}]" if type(i) != str else i for i in labels]

    def diagram_distance(_):
        return squareform(distances)

    dendo = ff.create_dendrogram(
        model.labels_,
        labels=plotly_labels,
        colorscale=px.colors.qualitative.Plotly,
        distfun=diagram_distance,
        linkagefun=lambda x: linkage(x, params_json["linkage"]),
        color_threshold=params_json["dendrogram_cut"],
    )
    dendo.update_layout(hovermode="x")

    dendrogram_colors = set()
    for i in range(len(dendo["data"])):
        iterator = dendo["data"][i]
        fig.add_trace(iterator, row=len(dists) + 1, col=1)
        if i == 0:
            print(iterator)
        dendrogram_colors.add(iterator["marker"]["color"])

    print(len(dendo["data"]), len(model.labels_))
    print(list(dendrogram_colors))

    # fig.update_layout(
    #     width=1500,
    #     height=500,
    #     template="simple_white",
    #     showlegend=False,
    #     font=dict(color="black", size=10),
    #     title="Persistence Based Clustering",
    # )

    # fig.update_xaxes(
    #     dict(
    #         title=f"Persistence Diagrams of {params_json['projector'].upper()} Embeddings"
    #     )
    # )
    # fig.update_yaxes(dict(title=f"{metric} distance"))
    return fig


def save_visualizations_as_html(visualizations, output_file):
    """
    Saves a list of Plotly visualizations as an HTML file.

    Parameters:
    -----------
    visualizations : list
        A list of Plotly visualizations (plotly.graph_objects.Figure).
    output_file : str
        The path to the output HTML file.
    """

    # Create the HTML file and save the visualizations

    with open(output_file, "w") as f:
        f.write("<html>\n<head>\n</head>\n<body>\n")
        for i, viz in enumerate(visualizations):
            div_str = pio.to_html(viz, full_html=False, include_plotlyjs="cdn")
            f.write(f'<div id="visualization{i+1}">{div_str}</div>\n')
        f.write("</body>\n</html>")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src/")
    import data
    from utils import convert_to_gtda, get_diagrams

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
        "-m",
        "--metric",
        type=str,
        default=params_json["diagram_metric"],
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )
    parser.add_argument(
        "-l",
        "--linkage",
        type=str,
        default=params_json["linkage"],
        help="Select linkage algorithm for building Agglomerative Clustering Model.",
    )
    parser.add_argument(
        "-c",
        "--dendrogram_cut",
        type=str,
        default=params_json["dendrogram_cut"],
        help="Select metric (that is supported by Giotto) to compare persistence daigrams.",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        default=params_json["normalize"],
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

    projections_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/projections/"
        + params_json["projector"]
        + "/",
    )
    diagrams_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/diagrams/"
        + params_json["projector"]
        + "/",
    )
    distances_in_file = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/EQC/"
        + params_json["projector"]
        + "/distance_matrices/"
        + f"{metric}_pairwise_distances.pkl",
    )
    model_in_file = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/EQC/"
        + params_json["projector"]
        + "/models/"
        + f"embedding_clustering_{metric}_{args.linkage}-linkage_{args.dendrogram_cut}.pkl",
    )

    with open(distances_in_file, "rb") as D:
        reference = pickle.load(D)

    keys, distances = reference["keys"], reference["distances"]

    with open(model_in_file, "rb") as M:
        model = pickle.load(M)["model"]

    # cluster_labels = list(model.labels_)
    # N = len(cluster_labels)
    # tokens = {}
    # for label in cluster_labels:
    #     mask = np.where(cluster_labels == label, True, False)
    #     idxs = np.array(range(N))[mask]
    #     # Min Distance from original space
    #     i = np.argmin(distances[original][idxs])
    #     tokens[label] = keys[i]

    # generator = getattr(data, params_json["data_set"])
    # X, C, _ = generator(N=params_json["num_samples"])

    projection_figure = visualize_clustered_umaps(
        projections_dir,
        model=model,
        keys=keys,
        distances=distances,
    )

    # plotly_dendo = visualize_dendrogram(
    #     labels=plotly_labels, distances=distances, metric=metric
    # )

    # data_fig = visualize_data(X, colors)

    out_file = "equivalence_classes.html"
    out_dir = os.path.join(root, "data/" + params_json["data_set"])

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_file)
    save_visualizations_as_html([projection_figure], out_file)