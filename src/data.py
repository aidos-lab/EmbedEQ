"""Data set generator methods."""

import numpy as np
from sklearn.datasets import make_swiss_roll


def swiss_roll(N: int = 1500, hole: bool = False, **kwargs):
    """Generate Swiss Roll data set."""

    data, color = make_swiss_roll(
        n_samples=N,
        random_state=0,
        hole=hole,
    )
    return data, color
