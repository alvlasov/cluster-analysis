import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_cluster(n_dim, n_objects, intermix_factor, d1=-1, d2=1):
    a1 = 0.025
    a2 = 0.05

    mean = intermix_factor * (d1 + (d2 - d1) * np.random.rand(n_dim))
    #     print(mean)
    cov_diag = ((d2 - d1) * (a1 + (a2 - a1) * np.random.rand(n_dim))) ** 2
    #     print(cov_diag)
    cov = np.diag(cov_diag)
    sample = np.random.multivariate_normal(mean, cov, n_objects)

    return sample


def plot_principal_components(data, return_pca=False, plt_args={}):
    pca = PCA(n_components=2)
    data_2dim = pca.fit_transform(data)
    plt.scatter(*data_2dim.T, **plt_args)
    if return_pca:
        return pca


def get_principal_components(data, return_pca=False):
    pca = PCA(n_components=2)
    data_2dim = pca.fit_transform(data)
    if return_pca:
        return data_2dim, pca
    else:
        return data_2dim


def generate_noise(sample, percent):
    n_objects, n_dim = sample.shape
    n_noise_objects = int(n_objects * percent)
    max_ = np.max(sample, axis=0)
    min_ = np.min(sample, axis=0)
    noise = np.random.rand(n_noise_objects, n_dim) * (max_ - min_) + min_

    return noise


def generate_cluster_sizes(n_objects, n_clusters, cluster_min_size):
    n_free_objects = n_objects - n_clusters * cluster_min_size

    r1 = np.concatenate((np.zeros(1), np.random.rand(n_clusters - 1)))
    r2 = np.diff(np.sort(r1))

    n_extra_obj_in_cluster = np.round(r2 * n_free_objects)
    n_extra_obj_in_cluster = np.concatenate((np.ones(1) * (n_free_objects - n_extra_obj_in_cluster.sum()),
                                             n_extra_obj_in_cluster))

    n_obj_in_cluster = (n_extra_obj_in_cluster + cluster_min_size).astype(int)

    return n_obj_in_cluster


def generate_dataset(n_objects, n_clusters, n_dim, intermix_factor, cluster_min_size):
    # print('Generating dataset. Obj=%d, Clust=%d, Dim=%d, Intermix=%.3f, Clust_min_size=%d' % (n_objects, n_clusters,
    # n_dim, intermix_factor, cluster_min_size))

    clusters_size = generate_cluster_sizes(n_objects, n_clusters, cluster_min_size)
    # print('Cluster sizes = %s' % clusters_size)

    data = []
    labels = []

    for n, cl_size in enumerate(clusters_size):
        data.append(generate_cluster(n_dim, cl_size, intermix_factor))
        labels.append(np.full(cl_size, n))

    data = np.concatenate(tuple(data))
    labels = np.concatenate(tuple(labels))

    return data, labels
