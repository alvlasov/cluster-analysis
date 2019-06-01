from collections import namedtuple

import numpy as np
from sklearn.cluster import k_means

from .ward import ward_clustering


def ikmeansb(X: np.array, min_size=1, min_scatter_rate=0, n_clusters=None):
    n, m = X.shape

    max_ = np.max(X, axis=0)
    min_ = np.min(X, axis=0)
    mean = np.mean(X, axis=0)
    range_ = max_ - min_

    sY = (X - mean) / range_
    D = np.sum(sY ** 2)

    ancl = anomalous(X, mean, range_, D)
    # pprint('anomalous clusters:')
    # pprint([(i+1, len(x.indices), x.scatter_rate) for i, x in enumerate(ancl)])
    if n_clusters is None:
        ancl = [x for x in ancl if len(x.indices) > min_size and x.scatter_rate > min_scatter_rate]

    ac_centers = np.vstack([X[x.indices].mean(axis=0) for x in ancl])
    ac_labels = np.full(n, -1)
    for i, x in enumerate(ancl):
        ac_labels[x.indices] = i

    if n_clusters is not None:
        ac_reduced_labels = ward_clustering(X, ac_labels, stopping_criterion='n_clusters', n_clusters=n_clusters)
        ac_centers = np.array([X[ac_reduced_labels == i].mean(axis=0) for i in np.unique(ac_reduced_labels)])

    X_labels, centrestand, centrereal, wd = k_means_(sY, ac_centers, range_, mean)

    return X_labels


def ikmeansb_sklearn(X: np.array, min_size=1, min_scatter_rate=0, n_clusters=None):
    n, m = X.shape

    max_ = np.max(X, axis=0)
    min_ = np.min(X, axis=0)
    mean = np.mean(X, axis=0)
    range_ = max_ - min_

    sY = (X - mean) / range_
    D = np.sum(sY ** 2)

    ancl = anomalous(X, mean, range_, D)
    # pprint('anomalous clusters:')
    # pprint([(i+1, len(x.indices), x.scatter_rate) for i, x in enumerate(ancl)])
    if n_clusters is None:
        ancl = [x for x in ancl if len(x.indices) > min_size and x.scatter_rate > min_scatter_rate]

    ac_centers = np.vstack([X[x.indices].mean(axis=0) for x in ancl])
    ac_labels = np.full(n, -1)
    for i, x in enumerate(ancl):
        ac_labels[x.indices] = i

    if n_clusters is not None:
        ac_reduced_labels = ward_clustering(X, ac_labels, stopping_criterion='n_clusters', n_clusters=n_clusters)
        ac_centers = np.array([X[ac_reduced_labels == i].mean(axis=0) for i in np.unique(ac_reduced_labels)])

    _, X_labels, _ = k_means((X - mean) / range_, n_clusters=len(ac_centers),
                             init=ac_centers,
                             n_init=1,
                             verbose=False,
                             algorithm='full')

    return X_labels


Cluster = namedtuple('Cluster', ['indices', 'centroid', 'scatter_rate'])


def anomalous(X, mean, range_, D, noprint=True):
    if noprint: print = lambda *x: None

    n, m = X.shape

    remains = np.arange(n)
    clusters = []
    while len(remains) > 0:

        distance = dist(X, remains, range_, mean)
        ind = np.argmax(distance)
        index = remains[ind]
        centroid = X[index]
        print(centroid)

        flag = True
        while flag:

            dist_a = dist(X, remains, range_, centroid)
            dist_b = dist(X, remains, range_, mean)
            clus = np.argwhere(dist_a < dist_b).flatten()
            # print(clus)
            cluster = remains[clus]

            if len(cluster) > 0:
                new_center = X[cluster].mean(axis=0)
            else:
                new_center = centroid
            if not np.allclose(centroid, new_center):
                centroid = new_center
            else:
                flag = False
        print(len(cluster), centroid, '\n')

        censtand = (centroid - mean) / range_
        dD = np.sum(censtand ** 2) * len(cluster) / D
        clusters.append(Cluster(cluster, centroid, dD))
        remains = np.setdiff1d(remains, cluster)

    return clusters


def dist(X, remains, range_, a):
    """ finding normalized distances in 'remains' to point 'a'"""
    Y = (X[remains] - a)  # / range_ # Отличие
    return np.sum(Y ** 2, axis=1)


def k_means_(Y, cent, range_, me):
    flag = 0
    dd = np.sum(Y ** 2, axis=1)

    membership = np.zeros(len(Y))
    while flag == 0:
        labelc, wc = clusterupdate(Y, cent)
        if np.array_equal(labelc, membership):
            flag = 1
            centre = cent
            w = wc
        else:
            cent = ceupdate(Y, labelc)
            membership = labelc

    centrestand = centre
    centrereal = centre * range_ + me

    wd = w * 100 / dd
    return membership, centrestand, centrereal, wd


def clusterupdate(Y, cent):
    disto = []
    for k in range(len(cent)):
        cc = cent[k]
        dif = Y - cc
        disto.append(np.sum(dif ** 2, axis=1))
    disto = np.vstack(disto).T
    labelc = np.argmin(disto, axis=1)
    aa = np.min(disto, axis=1)
    wc = np.sum(aa)
    return labelc, wc


def ceupdate(X, labelc):
    centres = []
    for kk in np.unique(labelc):
        clk = np.argwhere(labelc == kk).flatten()
        elemk = X[clk]
        centres.append(np.mean(elemk, axis=0))
    centres = np.vstack(centres)
    return centres
