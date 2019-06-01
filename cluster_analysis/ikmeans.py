import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import k_means

from .utils.dataset_generator import plot_principal_components
from .ward import ward_clustering

warnings.simplefilter('ignore')

global_pca = PCA(2)


def anomalous_cluster(X, center=True, plot=False):
    if center:
        zero = X.mean(axis=0)
    else:
        zero = 0
    X_c = X - zero

    global global_pca

    distances = np.sum((X - zero) ** 2, axis=1)
    anomalous_center = X[np.argmax(distances)]

    if plot:
        # plt.scatter(*global_pca.transform(X).T)
        plt.scatter(*global_pca.transform(anomalous_center[None]).T, c='g', marker='+', zorder=25, s=250)

    while True:

        dist_to_anomalous = np.linalg.norm(X - anomalous_center, axis=1)
        dist_to_zero = np.linalg.norm(X - zero, axis=1)

        mask = (dist_to_anomalous < dist_to_zero)

        new_anomalous_center = X[mask].mean(axis=0)

        if np.array_equal(new_anomalous_center, anomalous_center):
            break

        anomalous_center = new_anomalous_center

        if plot: plt.scatter(*global_pca.transform(anomalous_center[None]).T, c='g', marker='*', zorder=25, s=250)

    #     if plot: plt.show()

    anomalous_cluster_idx = np.argwhere(mask).flatten()
    return anomalous_cluster_idx, anomalous_center + zero


def ikmeans(X, t=1, n_clusters=None, plot=False):
    zero = X.mean(axis=0)
    X_c = X - zero

    global global_pca
    global_pca.fit_transform(X_c)

    obj_set = np.full(len(X), True)

    ac_objects = []
    ac_centers = []

    n = 1
    while np.any(obj_set):

        ac_obj, ac_center = anomalous_cluster(X_c[obj_set], center=False, plot=True)

        if plot:
            if np.any(obj_set): plt.scatter(*global_pca.transform(X_c[obj_set]).T, c='grey')
            if np.any(~obj_set): plt.scatter(*global_pca.transform(X_c[~obj_set]).T, c='lightgrey')
            plt.scatter(*global_pca.transform([[0], [0]]).T, c='k', zorder=10)
            plt.scatter(*global_pca.transform(ac_center[None]).T, c='y', zorder=10)
            plt.show()

        selected_objects_id = np.argwhere(obj_set).flatten()
        ac_obj = selected_objects_id[ac_obj]

        ac_objects.append(ac_obj)
        ac_centers.append(ac_center + zero)

        obj_set[ac_obj] = False

        n += 1

    ac_centers = np.array([i for i, j in zip(ac_centers, ac_objects) if len(j) > t])
    ac_labels = np.ones(len(X)) * (-1)
    for i, objs in enumerate(ac_objects):
        ac_labels[objs] = i

    if n_clusters is not None:
        ac_reduced_labels = ward_clustering(X, ac_labels, stopping_criterion='n_clusters', n_clusters=n_clusters)
        ac_centers = np.array([X[ac_reduced_labels == i].mean(axis=0) for i in np.unique(ac_reduced_labels)])

    _, X_labels, _ = k_means(X, n_clusters=len(ac_centers),
                             init=ac_centers,
                             n_init=1,
                             verbose=False,
                             algorithm='full')
    return X_labels


def extract_anomalous_centers(X, t=1):
    zero = X.mean(axis=0)
    X_c = X - zero

    obj_set = np.full(len(X), True)

    ac_objects = []
    ac_centers = []

    n = 1
    while np.any(obj_set):
        ac_obj, ac_center = anomalous_cluster(X_c[obj_set], center=False)

        selected_objects_id = np.argwhere(obj_set).flatten()
        ac_obj = selected_objects_id[ac_obj]

        ac_objects.append(ac_obj)
        ac_centers.append(ac_center + zero)

        obj_set[ac_obj] = False

        n += 1

    ac_centers = np.array([i for i, j in zip(ac_centers, ac_objects) if len(j) > t])

    return ac_centers


def ikmeans_wd(X):
    ''' Post iKMeans Ward clustering with TTP stopping criterion '''

    X_labels = ikmeans(X)
    X_labels = ward_clustering(X, initial_clusters=X_labels, stopping_criterion='ttp')

    return X_labels


def ikmeans_wt(X):
    ''' Post iKMeans Ward clustering with Threshold stopping criterion '''

    X_labels = ikmeans(X)
    X_labels = ward_clustering(X, initial_clusters=X_labels, stopping_criterion='threshold', threshold_alpha=0.5)

    return X_labels
