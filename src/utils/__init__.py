import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

#  ╭──────────────────────────────────────────────────────────╮
#  │ Utility Functions                                        │
#  ╰──────────────────────────────────────────────────────────╯


def assign_labels(data, n_clusters, k=10):
    """Assign Labels for Sklearn Data Sets based on KNN Graphs"""

    connectivity = kneighbors_graph(data, n_neighbors=k, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    ).fit(data)
    labels = ward.labels_
    return labels


def gtda_pad(diagrams, dims=(0, 1)):
    homology_dims = {}
    sizes = {}
    for i, diagram in enumerate(diagrams):
        tmp = {}
        counter = {}
        for dim in dims:
            # Generate Sub Diagram for particular dim
            sub_dgm = diagram[diagram[:, 2] == dim]
            counter[dim] = len(sub_dgm)
            tmp[dim] = sub_dgm

        homology_dims[i] = tmp
        sizes[i] = counter

    # Building Padded Diagram Template
    total_features = 0
    template_sizes = {}
    for dim in dims:
        size = max([dgm_id[dim] for dgm_id in sizes.values()])
        template_sizes[dim] = size
        total_features += size

    template = np.zeros(
        (
            len(diagrams),
            total_features,
            3,
        )
    )
    # Populate Template
    for i in range(len(diagrams)):
        pos = 0  # position in template
        for dim in dims:
            original_len = pos + sizes[i][dim]
            template_len = pos + template_sizes[dim]
            template[i, pos:original_len, :] = homology_dims[i][dim]

            template[i, pos:template_len, 2] = int(dim)
            # Reset position for next dimension
            pos += template_sizes[dim]

    return template


def format_arguments(args: list):
    return (arg.strip() for arg in args)
