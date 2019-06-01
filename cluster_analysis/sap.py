from functools import partial
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

from .affinity_propagation import affinity_propagation, similarities_euclid


def _run_sap(data, sim, p_i, n_run):
    n_points = sim.shape[0]
    perturbed_labels_all = []
    for i in range(n_run):

        perturbed_idx = np.random.choice(range(n_points), size=int(0.8 * n_points), replace=False)
        perturbed_mask = np.zeros(n_points, dtype=bool)
        perturbed_mask[perturbed_idx] = True

        perturbed_sim = sim[perturbed_idx][:, perturbed_idx]
        perturbed_sim[np.eye(perturbed_sim.shape[0], dtype=bool)] = p_i

        perturbed_labels = affinity_propagation(perturbed_sim, damping_factor=0.5)
        exemplars = np.unique(perturbed_labels)

        rem_labels = -1 * np.ones((~perturbed_mask).sum())
        for n, i in enumerate(data[~perturbed_mask]):
            rem_labels[n] = exemplars[np.square(data[exemplars] - i).sum(axis=1).argmin()]

        all_labels = -1 * np.ones(n_points)
        all_labels[perturbed_mask] = perturbed_labels
        all_labels[~perturbed_mask] = rem_labels
        label_dict = {j: i for i, j in enumerate(exemplars)}
        all_labels = np.vectorize(label_dict.get)(all_labels)

        perturbed_labels_all.append(all_labels)

    permuted_labels_all = [np.random.permutation(x) for x in perturbed_labels_all]

    scores = []
    scores_permuted = []
    for i, j in combinations(range(n_run), 2):
        scores.append(normalized_mutual_info_score(perturbed_labels_all[i], perturbed_labels_all[j]))
        scores_permuted.append(normalized_mutual_info_score(permuted_labels_all[i], permuted_labels_all[j]))

    return p_i, np.mean(scores) / np.mean(scores_permuted)


def stability_affinity_propagation(data, n_pref, n_run, parallel=False):
    sim = similarities_euclid(data)
    n_points = sim.shape[0]

    p_min = sim.min()
    p_max = sim.max()

    ps = [p_min + (p_max - p_min) / (n_pref - 1) * i for i in range(n_pref)]

    _run = partial(_run_sap, data, sim, n_run=n_run)

    if not parallel:
        scores_p_normalized = []
        for p_i in tqdm(ps):
            _, score_norm = _run(p_i)
            scores_p_normalized.append(score_norm)
    else:
        pool = Pool()
        result = {}
        for p_i, score in tqdm(pool.imap_unordered(_run, ps), total=n_pref):
            result[p_i] = score
        scores_p_normalized = [result[i] for i in ps]

    plt.plot(ps, scores_p_normalized)
    best_p = ps[np.argmax(scores_p_normalized)]
    print('Best p =', best_p)
    sim[np.eye(sim.shape[0], dtype=bool)] = best_p
    pred_labels = affinity_propagation(sim, damping_factor=0.5)
    return pred_labels
