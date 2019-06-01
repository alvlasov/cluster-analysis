import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from .affinity_propagation import affinity_propagation, similarities_mirkin, similarities_euclid
from .ward import ward_clustering
from .ikmeans import ikmeans

warnings.simplefilter('ignore')


def c4(X, similarities='mirkin', alpha=0.5, normalize=True, plot=False, verbose=0):
    X_c = X - X.mean(axis=0)

    if similarities == 'mirkin':
        sim = similarities_mirkin(X_c, normalize=normalize)
    elif similarities == 'euclid':
        sim = similarities_euclid(X_c)
    else:
        raise ValueError

    # AP
    labels_ap = affinity_propagation(sim)
    exemplars = np.unique(labels_ap)
    centers = X[exemplars]
    if plot: plt.scatter(*centers.T, c='r', s=150, zorder=9)
    print('After AP: %d clusters' % len(exemplars))

    # KM1
    km = KMeans(n_clusters=len(centers), init=centers)
    labels_km_1 = km.fit_predict(X)
    print('After KM1: %d clusters' % len(np.unique(labels_km_1)))

    # Ward
    labels_ward = ward_clustering(X, initial_clusters=labels_km_1, stopping_criterion='threshold',
                                  threshold_alpha=alpha,
                                  verbose=verbose)

    # One-hot encoding
    unique_labels = np.unique(labels_ward)
    n_values = np.max(unique_labels) + 1
    labels = np.eye(n_values, dtype=bool)[labels_ward].T
    centers_ward = np.array([np.mean(X[x], axis=0) for x in labels])
    print('After Ward: %d clusters' % len(unique_labels))

    if plot: plt.scatter(*centers_ward.T, c='k', s=150, zorder=10)

    # KM2
    km = KMeans(n_clusters=len(centers_ward), init=centers_ward)
    labels_km_2 = km.fit_predict(X)
    print('After KM2: %d clusters' % len(np.unique(labels_km_2)))

    return labels_km_2


def c4_anomalous(X, similarities='mirkin', alpha=0.5, plot=False, verbose=0):
    # Anomalous clusters
    labels_ikm = ikmeans(X)
    centers = [X[labels_ikm == i].mean(axis=0) for i in np.unique(labels_ikm)]
    print('After iKM: %d clusters' % len(centers))

    # Affinity propagation
    preferences = np.array([1.0 / (0.01 + np.sqrt(np.square(centers[l] - x).sum())) for x, l in zip(X, labels_ikm)])

    X_c = X - X.mean(axis=0)

    sim = X_c.dot(X_c.T)

    di = np.diag_indices(X.shape[0])
    sim[di] = preferences

    labels_ap = affinity_propagation(sim)
    exemplars = np.unique(labels_ap)
    centers = X[exemplars]
    if plot: plt.scatter(*centers.T, c='r', s=150, zorder=9)
    print('After AP: %d clusters' % len(exemplars))

    # KM1
    km = KMeans(n_clusters=len(centers), init=centers)
    labels_km_1 = km.fit_predict(X)
    print('After KM1: %d clusters' % len(np.unique(labels_km_1)))

    # Ward
    labels_ward = ward_clustering(X, initial_clusters=labels_km_1, stopping_criterion='threshold',
                                  threshold_alpha=alpha,
                                  verbose=verbose)

    # One-hot encoding
    unique_labels = np.unique(labels_ward)
    n_values = np.max(unique_labels) + 1
    labels = np.eye(n_values, dtype=bool)[labels_ward].T
    centers_ward = np.array([np.mean(X[x], axis=0) for x in labels])
    print('After Ward: %d clusters' % len(unique_labels))

    if plot: plt.scatter(*centers_ward.T, c='k', s=150, zorder=10)

    # KM2
    km = KMeans(n_clusters=len(centers_ward), init=centers_ward)
    labels_km_2 = km.fit_predict(X)
    print('After KM2: %d clusters' % len(np.unique(labels_km_2)))

    return labels_km_2
