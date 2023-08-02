"Check the way diagrams are being read in when computing distance. Probably having the original space read in diagram -1"
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


def visualize_token_umaps(dir, tokens, dendrogram_colors):
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
    hashmap, neighbors, dists, coords = subplot_grid(dir)
    num_rows = 1
    num_cols = len(tokens)

    column_labels = [str(x[:2]) for x in tokens.values()]

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        column_titles=column_labels,
        x_title="Token Embeddings",
    )
    coords = [x[:2] for x in tokens.values()]
    col = 1
    for coord in coords:
        ref = str(coord).replace(" ", "")
        proj_2d = hashmap[ref]

        color = dendrogram_colors[ref]
        df = pd.DataFrame(proj_2d, columns=["x", "y"])

        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color,
                ),
            ),
            row=1,
            col=col,
        )
        col += 1

    fig.update_layout(
        width=1500,
        height=500,
        template="simple_white",
        showlegend=True,
        font=dict(color="black"),
        title="Token Embeddings",
    )

    fig.update_xaxes(
        showticklabels=False,
        tickwidth=0,
        tickcolor="rgba(0,0,0,0)",
        categoryorder="category ascending",
    )
    fig.update_yaxes(
        showticklabels=False,
        tickwidth=0,
        tickcolor="rgba(0,0,0,0)",
    )
    return fig


def visualize_clustered_umaps(dir, keys, dendrogram_colors, id_="original space"):
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
    hashmap, neighbors, dists, coords = subplot_grid(dir)
    num_rows = len(dists)
    num_cols = len(neighbors)
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        column_titles=list(map(str, neighbors)),
        x_title="n_neighbors",
        row_titles=list(map(str, dists)),
        y_title="min_dist",
    )
    keys.remove(id_)
    row = 1
    col = 1
    for coord in coords:
        ref = str(coord).replace(" ", "")
        proj_2d = hashmap[ref]

        color = dendrogram_colors[ref]
        df = pd.DataFrame(proj_2d, columns=["x", "y"])

        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color,
                ),
            ),
            row=row,
            col=col,
        )
        row += 1
        if row == len(dists) + 1:
            row = 1
            col += 1

    fig.update_layout(
        width=1500,
        template="simple_white",
        showlegend=False,
        font=dict(color="black"),
        title="Projection Gridsearch Plot",
    )

    fig.update_xaxes(
        showticklabels=False,
        tickwidth=0,
        tickcolor="rgba(0,0,0,0)",
        categoryorder="category ascending",
    )
    fig.update_yaxes(
        showticklabels=False,
        tickwidth=0,
        tickcolor="rgba(0,0,0,0)",
    )
    return fig


def visualize_dendrogram(labels, distances, metric):
    def diagram_distance(_):
        return squareform(distances)

    fig = ff.create_dendrogram(
        np.arange(len(labels)),
        labels=labels,
        distfun=diagram_distance,
        colorscale=px.colors.qualitative.Plotly,
        linkagefun=lambda x: linkage(x, params_json["linkage"]),
        color_threshold=params_json["dendrogram_cut"],
    )
    fig.update_layout(
        width=1500,
        height=1000,
        template="simple_white",
        showlegend=False,
        font=dict(color="black", size=10),
        title="Persistence Based Clustering",
    )

    fig.update_xaxes(title=dict(text=f"Embedding Parameters"))
    fig.update_yaxes(title=dict(text=f"{metric} homological distance"))

    ticktext = fig["layout"]["xaxis"]["ticktext"]
    tickvals = fig["layout"]["xaxis"]["tickvals"]
    colormap = {}
    reference = dict(zip(tickvals, ticktext))

    # Extracting Dendrogram Colors
    for trace in fig["data"]:
        if 0 in trace["y"]:
            xs = trace["x"][np.argwhere(trace["y"] == 0)]
            # This catch will ensure plots are generated, but empty plots may indicate you
            # have mismatched info between old runs and your params.json. Clean and rerun
            tickers = [reference[x[0]] if x[0] in reference.keys() else 0 for x in xs]
            for ticker in tickers:
                colormap[ticker] = trace["marker"]["color"]
    return fig, colormap


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
    from utils import embedding_coloring, subplot_grid

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
    else:
        metric = args.metric

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

    selection_in_file = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/EQC/"
        + params_json["projector"]
        + "/selected_embeddings/"
        + f"selection_{metric}_{args.linkage}-linkage_{args.dendrogram_cut}.pkl",
    )

    with open(distances_in_file, "rb") as D:
        reference = pickle.load(D)

    keys, distances = reference["keys"], reference["distances"]
    with open(model_in_file, "rb") as M:
        model = pickle.load(M)["model"]

    with open(selection_in_file, "rb") as s:
        tokens = pickle.load(s)

    labels = []
    for i, key in enumerate(keys):
        if type(key) == str:
            labels.append(key)
            id_ = key
        else:
            labels.append(key[:2])

    # DENDROGRAM
    plotly_labels = [f"[{i[0]},{i[1]}]" if type(i) != str else i for i in labels]

    plotly_dendo, colormap = visualize_dendrogram(
        labels=plotly_labels,
        distances=distances,
        metric=metric,
    )

    # CLUSTERED PROJECTIONS
    projection_figure = visualize_clustered_umaps(
        projections_dir, keys=keys, dendrogram_colors=colormap, id_=id_
    )

    # SELECTED EMBEDDINGS
    # New colors
    token_color_map = embedding_coloring(colormap)
    token_figure = visualize_token_umaps(projections_dir, tokens, token_color_map)
    out_file = f"{args.data}_equivalence_classes.html"
    out_dir = os.path.join(
        root,
        "data/" + params_json["data_set"] + "/synopsis/" + params_json["projector"],
    )

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_file)
    save_visualizations_as_html(
        [plotly_dendo, projection_figure, token_figure], out_file
    )
