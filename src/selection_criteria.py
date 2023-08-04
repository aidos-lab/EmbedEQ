"Functions to determine token embedding selection from an equivalency class"

import numpy as np


def min_topological_distance(keys, labels, distances, id_, **kwargs):
    # TODO: Select based on id_
    assert len(distances) > 0
    original_space_idx = [True if type(key) is str else False for key in keys]
    original_distances = distances[original_space_idx][0]
    selection = {}
    for label in np.unique(labels):
        mask = np.where(labels == label, True, False)
        class_distances = original_distances[mask]

        # If EQ class is > 1 and contains original space
        if mask[original_space_idx]:
            if len(class_distances) == 1:
                continue
            else:
                # Force selecting another embedding
                class_distances[class_distances == 0] = np.infty
        min_dist = np.min(class_distances)
        idx = np.where(original_distances == min_dist)[0][0]
        selection[label] = keys[idx]
    return selection
