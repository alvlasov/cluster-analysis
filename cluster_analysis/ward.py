import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.distance import minkowski
from tqdm import tqdm

from .depddp import _density_estimate, _pc_projection


def ward_clustering(X, initial_clusters=None, stopping_criterion='none', n_clusters=2, threshold_alpha=0.5, verbose=0):
    ''' 
    Ward agglomerative clustering algorithm.
    
    Parameters
    ----------
    X : array, shape=(n_samples, n_features)
        Training instances to cluster.
        
    initial_clusters : array, shape=(n_samples,) or None
        Initial partition of training samples. Contains cluster identifiers for every training instance.
        If None, use a trivial partition consisting of n_samples singletons.
        
    stopping_criterion : {'none', 'n_clusters', 'ttp', 'threshold'}
        A stopping condition for clustering process:
        
        'none' : stop when all objects are merged into single cluster
        'n_clusters' : stop when the number of clusters is equal to n_clusters
        'ttp' : merge only if the clusters are satisfying the TTP criterion, stop when no further mergers are possible
        'threshold' : merge only if the clusters are satisfying the Threshold criterion, stop when no further mergers are possible
    
    n_clusters : int
        Number of clusters for 'n_clusters' stopping criterion.
        
    threshold_alpha : float, in [0, 1]
        Parameter for 'threshold' stopping criterion.
        
    verbose : int, default 0
        If 0, no debug messages are printed.
        If >0, print debug messages.
        
        
    Returns
    -------    
    
    linkage : array, shape=(n_samples-1, 4)
        Linkage matrix for training set. Returned if stopping criterion is 'none'.
        
    labels : array, shape=(n_samples,)
        Index of the cluster each sample belongs to. Returned in case of any other stopping criterion.
        
    '''

    # Define verbose print function
    verbose_print = print if verbose else lambda x: None

    # Define Ward distance function
    def _dist(i, j):
        d = np.power(centers[i] - centers[j], 2).sum()
        n_i = n_cluster_size[i]
        n_j = n_cluster_size[j]
        return d * n_i * n_j / (n_i + n_j)

    # Set initial clustering
    if initial_clusters is None:
        clusters_id = np.arange(X.shape[0])
        X_labels = clusters_id
        centers = X.copy()
        n_cluster_size = np.ones(len(centers))
    else:
        clusters = np.unique(initial_clusters)
        clusters_id = np.arange(len(clusters))
        X_labels = initial_clusters.copy()
        cluster_to_obj = {i: np.argwhere(initial_clusters == cl).flatten() for i, cl in enumerate(clusters)}
        centers = np.array([X[cluster_to_obj[i]].mean(axis=0) for i in clusters_id])
        n_cluster_size = np.array([len(cluster_to_obj[i]) for i in clusters_id])

    n_obj, n_dim = centers.shape

    if verbose: pbar = tqdm(total=n_obj)

    # Matrix of squared distance between clusters 
    dist = np.full((n_obj, n_obj), float('inf'))
    for i, j in zip(*np.triu_indices(n_obj, k=1)):
        dist[i, j] = _dist(i, j)

    # Init linkage matrix if there is no stopping criterion
    if stopping_criterion == 'none':
        clusters_id_linkage = clusters_id.copy()
        linkage = np.zeros((n_obj - 1, 4))

    if verbose > 2: verbose_print('Orig labels = %s' % X_labels)
    verbose_print('Labels = %s' % clusters_id)
    if verbose > 1: verbose_print('Centers:\n%s' % centers)
    verbose_print('Cluster sizes = %s' % n_cluster_size)
    if verbose > 1: verbose_print('Criterion increase matrix:\n%s' % dist)

    n_iter = 0
    saved_dist_list = []
    if verbose: pbar.update()
    while True:

        # Check stopping criterion
        if stopping_criterion == 'n_clusters':
            if np.unique(clusters_id).shape[0] <= n_clusters:
                break
        elif stopping_criterion in ['ttp', 'none', 'threshold']:
            if np.all(np.isinf(dist)):
                break
        else:
            raise ValueError('Wrong stopping_criterion.')

        verbose_print('---- Iteration %d' % n_iter)

        # Find clusters to merge (to minimize ward distance)
        rem_id, del_id = np.unravel_index(np.argmin(dist), dist.shape)

        if stopping_criterion == 'ttp':
            # Check if the density function of resulting cluster is convex
            density = _density_estimate(_pc_projection(X[(X_labels == rem_id) + (X_labels == del_id)]))
            verbose_print('Density function: %s' % density)
            rel_min_indices = argrelextrema(density, np.less)[0]

            if len(rel_min_indices) > 0:
                saved_dist_list.append((rem_id, del_id, dist[rem_id, del_id]))
                dist[rem_id, del_id] = float('inf')
                verbose_print('Skipping merge of %d, %d, density function is not concave!' % (rem_id, del_id))
                verbose_print('Criterion increase matrix:\n%s' % dist)
                continue  # skip this merge
            else:
                for i, j, d in saved_dist_list:
                    dist[i, j] = d

                saved_dist_list.clear()

        elif stopping_criterion == 'threshold':
            # Check if the threshold condition on increase of the square loss criterion holds
            X_clust = X[(X_labels == rem_id) + (X_labels == del_id)]
            D = np.square(X_clust - X_clust.mean(axis=0)).sum()

            verbose_print('D = %.6f, d = %.6f' % (D, dist[rem_id, del_id]))

            if dist[rem_id, del_id] >= threshold_alpha * D and (
                    n_cluster_size[rem_id] > 1 or n_cluster_size[del_id] > 1):
                saved_dist_list.append((rem_id, del_id, dist[rem_id, del_id]))
                dist[rem_id, del_id] = float('inf')
                verbose_print('Skipping merge of %d, %d, threshold condition is violated!' % (rem_id, del_id))
                continue  # skip this merge
            else:
                for i, j, d in saved_dist_list:
                    dist[i, j] = d

                saved_dist_list.clear()

                # Calculate center of the new cluster
        n_d = n_cluster_size[del_id]
        n_r = n_cluster_size[rem_id]
        centers[rem_id] = (n_d * centers[del_id] + n_r * centers[rem_id]) / (n_d + n_r)
        centers[del_id] = float('nan')

        # Assign all points to the new cluster
        clusters_id[clusters_id == del_id] = rem_id
        X_labels[X_labels == del_id] = rem_id

        # Fill the next item of linkage matrix 
        if stopping_criterion == 'none':
            linkage[n_iter, :] = [clusters_id_linkage[rem_id], clusters_id_linkage[del_id],
                                  np.sqrt(dist[rem_id, del_id]), n_cluster_size[rem_id] + n_cluster_size[del_id]]

            clusters_id_linkage[clusters_id == del_id] = n_obj + n_iter
            clusters_id_linkage[clusters_id == rem_id] = n_obj + n_iter

            # Set new cluster sizes
        n_cluster_size[rem_id] += n_cluster_size[del_id]
        n_cluster_size[del_id] = 0

        # Update ward distances between clusters
        dist[del_id, del_id:] = float('inf')
        dist[:del_id, del_id] = float('inf')

        for j in range(rem_id + 1, n_obj):
            if np.isinf(dist[rem_id, j]): continue
            dist[rem_id, j] = _dist(rem_id, j)

        for i in range(rem_id):
            if np.isinf(dist[i, rem_id]): continue
            dist[i, rem_id] = _dist(i, rem_id)

        # Increase iteration number & print some verbose info
        n_iter += 1
        if verbose: pbar.update()

        verbose_print('Combining clusters %d, %d' % (rem_id, del_id))
        if verbose > 2: verbose_print('Orig labels = %s' % X_labels)
        verbose_print('Labels = %s' % clusters_id)
        if stopping_criterion == 'none':
            verbose_print('Linkage labels = %s' % clusters_id_linkage)
        if verbose > 1: verbose_print('Centers:\n%s' % centers)
        verbose_print('Cluster sizes = %s' % n_cluster_size)
        if verbose > 1: verbose_print('Criterion increase matrix:\n%s' % dist)

        # Encode all cluster names
    result_clusters = np.sort(np.unique(X_labels))
    labels = {j: i for i, j in enumerate(result_clusters)}

    X_labels = np.vectorize(labels.get)(X_labels)

    if verbose: pbar.close()

    if stopping_criterion == 'none':
        return linkage
    else:
        return X_labels
