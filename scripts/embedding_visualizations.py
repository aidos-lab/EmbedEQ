import json
import os
import pickle
import sys


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from plotly.subplots import make_subplots
import logging


def visualize_umaps(dir, labels):
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

    row = 1
    col = 1
    for coord in coords:
        ref = str(coord).replace(" ", "")
        proj_2d = hashmap[ref]

        df = pd.DataFrame(proj_2d, columns=["x", "y"])
        df["labels"] = labels
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=df["labels"],
                    cmid=0.3,
                    colorscale="jet",
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
        # height=900,
        template="simple_white",
        showlegend=False,
        font=dict(color="black"),
        title="Projection Gridsearch Plot",
    )

    fig.update_xaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")
    fig.update_yaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")
    return fig


def visualize_data(X, labels):
    dim = X.shape[1]
    assert dim in [2, 3], "Only 2D and 3D Scatter Plots are supported"

    # 3D
    if dim == 3:
        df = pd.DataFrame(X, columns=["x", "y", "z"])
        df["labels"] = labels
        trace = go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(
                size=4,
                color=df["labels"],
                colorscale="jet",
            ),
        )
    else:
        df = pd.DataFrame(X, columns=["x", "y"])
        df["labels"] = labels
        trace = go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                size=8,
                color=df["labels"],
                colorscale="jet",
            ),
        )

    fig = go.Figure(data=trace)

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
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

    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src/")
    import data

    from utils import subplot_grid

    JSON_PATH = os.getenv("params")
    assert os.path.isfile(JSON_PATH), "Please configure .env to point to params.json"
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)

    in_dir = os.path.join(
        root,
        "data/"
        + params_json["data_set"]
        + "/projections/"
        + params_json["projector"]
        + "/",
    )

    generator = getattr(data, params_json["data_set"])
    logging.info(f"Using generator routine {generator}")
    X, labels = generator(
        N=params_json["num_samples"],
        random_state=params_json["random_state"],
        n_clusters=params_json["num_clusters"],
    )
    try:
        data_figure = visualize_data(X, labels)
    except AssertionError:
        data_figure = go.Figure()

    projection_figure = visualize_umaps(in_dir, labels)

    data_set = params_json["data_set"]
    out_file = f"{data_set}_embedding_summary.html"
    out_dir = os.path.join(
        root,
        "data/" + params_json["data_set"] + "/synopsis/" + params_json["projector"],
    )

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_file)

    save_visualizations_as_html([data_figure, projection_figure], out_file)
