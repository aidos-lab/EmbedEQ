import numpy as np
from utils import convert_to_gtda, get_diagrams
from scipy.spatial import distance_matrix
from gtda.diagrams import PairwiseDistance


def pairwise_distance(
    folder,
    metric="bottleneck",
):
    keys, diagrams = get_diagrams(folder)
    dgms = convert_to_gtda(diagrams)
    distance_metric = PairwiseDistance(metric=metric)
    distance_metric.fit(dgms)
    distances = distance_metric.transform(dgms)

    return keys, distances


def compute_magnitude(W, p=2, ts=np.arange(0.01, 5, 0.01)):
    """Computes the magnitude Mag(tX) for all t in the set ts.
    W - a matrix where each row is a point in R^n
            p - an integer the metric that should be used [1 is for l1 (Manhattan), 2 is for l2 (euclidean)]
            ts - an array of the values of t for which the magnitude should be computed"""
    dist_mtx = distance_matrix(W, W, p)
    if dist_mtx.shape[0] >= 1000:
        inv_fn = np.linalg.pinv
    else:
        inv_fn = np.linalg.inv
    Zs = []
    for t in ts:
        try:
            Z = inv_fn(np.exp(-t * dist_mtx))
            Zs.append(Z)
        except Exception as e:
            print(f"Exception: {e} for t: {t} perturbing matrix")
            D = np.exp(-t * dist_mtx) + 0.01 * np.identity(
                n=dist_mtx.shape[0]
            )  # perturb similarity mtx to invert
            Z = inv_fn(D)
            Zs.append(Z)
    magnitude = np.array([Z.sum() for Z in Zs])
    return magnitude
