def tSNE(data, hyperparams, seed=0, **kwargs):
    perplexity, ee, dim = hyperparams
    operator = TSNE(
        n_components=dim,
        perplexity=perplexity,
        early_exaggeration=ee,
    )
    projection = operator.fit_transform(data)
    return projection


def phate(data, hyperparams, seed=0, **kwargs):
    knn, gamma, metric, dim = hyperparams
    operator = PHATE(
        n_components=dim,
        knn=knn,
        gamma=gamma,
        knn_dist=metric,
        random_state=seed,
        verbose=0,
    )
    projection = operator.fit_transform(data)
    return projection


def isomap(data, hyperparams, seed=0, **kwargs):
    n, m, dim = hyperparams
    operator = Isomap(
        n_components=dim,
        n_neighbors=n,
        metric=m,
    )
    projection = operator.fit_transform(data)
    return projection


def LLE(data, hyperparams, seed=0, **kwargs):
    n, reg, dim = hyperparams
    operator = LocallyLinearEmbedding(
        n_components=dim,
        n_neighbors=n,
        reg=reg,
    )
    projection = operator.fit_transform(data)
    return projection
