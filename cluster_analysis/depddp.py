import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from anytree import Node, PreOrderIter

from .utils.dataset_generator import plot_principal_components


def _gaussian_density(x):
    return (2 * np.pi) ** (-0.5) * np.exp(-0.5 * x ** 2)


def _density_estimate(x, y=None):
    ''' Parzen type density estimation
        x - np.array '''

    y = x if y is None else y
    x = np.sort(x)
    values = []
    n = len(x)
    h = x.std() * (4 / 3 / n) ** (1 / 5)

    for i in y:
        f = 1 / n / h * _gaussian_density((x - i) / h).sum()
        values.append(f)

    return np.array(values)


def _pc_projection(x):
    ''' Perform projection of x on the first principal axis
        of x_centered. '''

    x_c = x - x.mean(axis=0)
    pc_vector = np.linalg.svd(x_c)[2][0]
    x_pc_unsorted = (np.dot(x_c, pc_vector))

    return x_pc_unsorted


def split_data(X, plot=False, labels='b'):
    ''' Performs splitting data into two leaves 
        according to dePDDP algorithm. '''

    X_c = X - X.mean(axis=0)

    pc_vector = np.linalg.svd(X_c)[2][0]

    X_pc_unsorted = (np.dot(X_c, pc_vector))
    # print(X_pc_unsorted)
    X_pc = np.sort(X_pc_unsorted)

    if plot:
        pc_vector2 = np.linalg.svd(X_c)[2][1]
        plt.scatter(np.dot(X_c, pc_vector), np.dot(X_c, pc_vector2))
        plt.show()

    # Calc density and find minima
    X_pc_density = _density_estimate(X_pc)
    rel_min_indices = argrelextrema(X_pc_density, np.less)[0]

    # If can not split, return
    if len(rel_min_indices) == 0:
        if plot:
            plt.scatter(X_pc, X_pc_density)
            plt.show()
        return

    # Determine best object for splitting
    best_idx = rel_min_indices[np.argmin(X_pc_density[rel_min_indices])]

    # Find split value on principal axis
    p = np.polyfit(X_pc[best_idx - 1:best_idx + 2], X_pc_density[best_idx - 1:best_idx + 2], deg=2)
    p_d = np.polyder(p)
    split_value = -p_d[1] / p_d[0]
    density_value = np.polyval(p, split_value)

    # Determine indices for left and right leaf
    left_split_idx_unsorted = (X_pc_unsorted < split_value)
    right_split_idx_unsorted = (X_pc_unsorted >= split_value)

    if plot:
        left_split_idx = (X_pc < split_value)
        right_split_idx = (X_pc >= split_value)
        plt.scatter(X_pc[left_split_idx], X_pc_density[left_split_idx], c='r')
        plt.scatter(X_pc[right_split_idx], X_pc_density[right_split_idx], c='g')
        plt.scatter(X_pc[best_idx], X_pc_density[best_idx], c='b')
        plt.scatter(split_value, density_value, c='pink')
        plt.show()

    return density_value, left_split_idx_unsorted, right_split_idx_unsorted


def depddp(X, return_tree=False, plot=False, verbose=0):
    if verbose:
        verbose_print = print
    else:
        verbose_print = lambda x: None

    indices = np.arange(X.shape[0])
    tree_root = Node('r', idx=indices)

    cur_nodes = [tree_root]

    while len(cur_nodes) > 0:

        verbose_print('Current nodes: %s' % [n.name for n in cur_nodes])
        splits = []
        for node in cur_nodes:
            verbose_print(' Node %s' % node.name)
            split = split_data(X[node.idx], plot=plot)
            if split is not None:
                d, left_idx, right_idx = split
                left_idx = node.idx[left_idx]
                right_idx = node.idx[right_idx]

                split = (d, left_idx, right_idx)
            splits.append(split)

        no_split_idx = [i for i, j in enumerate(splits) if j is None]

        cur_nodes = [j for i, j in enumerate(cur_nodes) if i not in no_split_idx]
        splits = [j for i, j in enumerate(splits) if i not in no_split_idx]

        if len(cur_nodes) > 0:
            min_dens_node_idx = min(enumerate(splits), key=lambda x: x[1][0])[0]

            splitted_node = cur_nodes.pop(min_dens_node_idx)
            name = splitted_node.name
            left_leaf = Node(name + '/l', parent=splitted_node,
                             idx=splits[min_dens_node_idx][1])
            right_leaf = Node(name + '/r', parent=splitted_node,
                              idx=splits[min_dens_node_idx][2])

            cur_nodes.append(left_leaf)
            cur_nodes.append(right_leaf)

    if not return_tree:
        i = 0
        for n in PreOrderIter(tree_root, filter_=lambda n: n.is_leaf):
            indices[n.idx] = i
            i += 1
        return indices
    else:
        return tree_root
