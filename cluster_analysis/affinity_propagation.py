import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def similarities_euclid(points):
    """  Compute similarities as minus euclidian distances. Preference is taken as median. """
    s_def = np.array([[-np.linalg.norm(i - j) ** 2 for i in points] for j in points])
    m = np.median(s_def[[~np.eye(s_def.shape[0], dtype=bool)]])
    for i in range(s_def.shape[0]):
        s_def[i, i] = m
    return s_def


def similarities_mirkin(points, normalize=False):
    """ Compute similarities using B. Mirkin method. """

    points_scaled = (points - points.mean(axis=0))

    if normalize:
        points_scaled /= points.std(axis=0)

    s_mir = points_scaled.dot(points_scaled.T)  # / len(points)
    return s_mir


def similarities_mirkin_norm_range(points):
    max_ = points.max(axis=0)
    min_ = points.min(axis=0)
    points_scaled = points - points.mean(axis=0)
    points_scaled /= (max_ - min_)

    s = points_scaled.dot(points_scaled.T)
    return s


def plot_ap(points, a, r, ex, alpha_threshold=0, alpha_threshold_arrows=0.95, figsize=(16, 9)):
    """ Plot fancy graph of points and relations between them. """

    plt.figure(figsize=figsize)
    for i, j in product(range(points.shape[0]), repeat=2):
        if i == j: continue
        a_r = a + r
        alpha_ = (a_r[i, j] - a_r.min()) / (a_r.max() - a_r.min())
        x1, y1 = points[i]
        x2, y2 = points[j]

        if alpha_ > alpha_threshold:
            plt.plot([x1, x2], [y1, y2], 'k', alpha=alpha_ ** 2, linewidth=3 * alpha_ ** 2, zorder=0)
            if alpha_ > alpha_threshold_arrows:
                dx = (x2 - x1) * alpha_
                dy = (y2 - y1) * alpha_
                plt.arrow(x1, y1, dx, dy, alpha=alpha_ ** 2, color='k', shape='full', lw=0,
                          length_includes_head=True, head_width=.05, zorder=0)

    plt.scatter(*points.T, c=ex, s=100, zorder=1)
    for i in np.unique(ex):
        plt.scatter(*points[i], color='r', s=300, zorder=0.5)
    plt.show()


def affinity_propagation(s, damping_factor=0.5, iter_threshold=10, max_iter=200, verbose=0, plot=False, points=None,
                         show_plot_after=5):
    """
    Affinity Propagation clustering. Uses some code from scikit-learn project.
    
    Parameters
    ----------
    
    s : array, shape=(n_samples,n_samples)
        Square matrix of similarities between training instances.
        
    damping_factor : float in [0.5, 1]
        Damping factor between 0.5 and 1.
        
    iter_threshold : int 
        Number of iterations with no change in exemplars 
        that stops the convergence.        
        
    verbose : int, default 0
        If > 0, print some debug messages.
        
    plot : {True, False}
        Plot all points and connections between them. Useful for small
        2-dimensional datasets.
        
    points : array, shape=(n_samples,n_features)
        Points to plot. Used only if plot == True.
        
    show_plot_after : int 
        How many iterations skip before drawing a graph.
        Used only if plot == True.
        
    Return
    ------
    
    ex : array, shape=(n_samples,)
        Array of exemplars for each training instance.
    
    """

    # Define verbose print function
    verbose_print = print if verbose else lambda x: None

    n_samples = s.shape[0]

    # Initialize messages
    a = np.zeros_like(s)
    r = np.zeros_like(s)

    tmp = np.zeros_like(s)
    ind = np.arange(n_samples)

    # Remove degeneracies
    s += ((np.finfo(np.double).eps * s + np.finfo(np.double).tiny * 100) *
          np.random.randn(n_samples, n_samples))

    n_iter_ex_unchanged = 0
    n_iter = 1

    while n_iter_ex_unchanged <= iter_threshold:

        # tmp = A + S; compute responsibilities
        np.add(a, s, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(s, Y[:, None], tmp)
        tmp[ind, I] = s[ind, I] - Y2

        # Damping
        tmp *= 1 - damping_factor
        r *= damping_factor
        r += tmp

        # tmp = Rp; compute availabilities
        np.maximum(r, 0, tmp)
        tmp.flat[::n_samples + 1] = r.flat[::n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping_factor
        a *= damping_factor
        a -= tmp

        new_ex = np.argmax(a + r, axis=1)
        if n_iter > 1:
            if np.array_equal(new_ex, ex):
                n_iter_ex_unchanged += 1
            else:
                n_iter_ex_unchanged = 0
        ex = new_ex

        if plot and verbose and (n_iter - 1) % show_plot_after == 0:
            verbose_print("Iteration %d" % n_iter)
            plot_ap(points, a, r, ex)

        n_iter += 1

        if n_iter >= max_iter:
            print('Not converged!')
            break

    verbose_print('Number of iterations = %d' % n_iter)

    return ex
