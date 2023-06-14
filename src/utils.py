"Utility functions."

import itertools


def parameter_coordinates(hyper_params: dict, embedding):

    assert embedding in ["umap", "tSNE"], f"{embedding} is not yet supported."
    if embedding == "umap":
        N = hyper_params["n_neighbors"]
        d = hyper_params["min_dist"]
        n = hyper_params["dim"]
        m = hyper_params["metric"]

        coordinates = list(itertools.product(N, d, n, m))

    return coordinates
