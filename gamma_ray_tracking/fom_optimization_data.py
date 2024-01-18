"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Data preparation for optimization of the FOM

Generates features for various permutations of the data for use in the
optimization process
"""
from itertools import permutations
from math import factorial
from typing import Dict, List, Tuple

import numpy as np

from gamma_ray_tracking.event_class import Event
from gamma_ray_tracking.event_tools import split_event_clusters
from gamma_ray_tracking.fom_tools import (
    get_all_features_cluster,
    individual_FOM_feature_names,
)

# from tqdm import tqdm


feature_names = list(individual_FOM_feature_names().keys())

m30_true_energies = {}  # true energies in MeV
for i in range(1, 31):
    m30_true_energies[i] = 0.08 + 0.09 * (i - 1)


def get_fom_pieces(event: Event, cluster: List, **kwargs):
    """Get the FOM values from the data"""
    return list(get_all_features_cluster(event, cluster, **kwargs).values())


def generate_semi_greedy_data(
    event: Event,
    true_clusters: Dict[int, List],
    true_energies: Dict[int, float],
    tol: float = 1e-2,
    max_cluster_size: int = 7,
    use_tango: bool = False,
    remove_pair_production: bool = True,
    width: int = 5,
    **kwargs,
) -> Tuple[List]:
    """
    Feature generation for single events. Get the features for the true order,
    then all other orders, and concatenate each order with the true order.

    Only creates data pairs between the true order and other orders.
    Does partial orders that would be used by a semi-greedy method.
    """
    features = []
    # y = []
    complete_energy = []
    energy_sums = []
    count = 0
    correct_solution_index = []
    other_solution_index = []
    lengths = []
    cluster_ids = []
    true_cluster_ids = []
    acceptable = []
    first_int_good = []
    optimal_order = []
    cluster_count = 1
    for cluster_id, cluster in true_clusters.items():
        if len(cluster) >= max_cluster_size or len(cluster) <= 1:
            continue

        if remove_pair_production:
            # remove pair production interactions and others; ordering doesn't make sense here
            if any(event.points[i].interaction_type > 2 for i in cluster):
                continue

        # l = factorial(len(cluster))
        l = 0
        energy_sum = sum(event.points[i].e for i in cluster)
        complete = abs(energy_sum - true_energies[cluster_id]) < tol
        ordered = True

        # Loop over all possible orders
        index1_start = count
        if width is None:
            cluster_width = None
        else:
            cluster_width = min(len(cluster), width)
            if (
                len(cluster) == width + 1
            ):  # special case where there is no savings for eliminating final element
                cluster_width = len(cluster)
        for pred_cluster in permutations(cluster, r=cluster_width):
            pred_cluster = list(pred_cluster)
            new_features = get_fom_pieces(
                event, pred_cluster, start_energy=energy_sum, **kwargs
            )
            features.append(new_features)
            l += 1
            correct_solution_index.append(index1_start)
            other_solution_index.append(count)
            count += 1
            # y.append([not ordered, not complete, not(ordered and complete)])
            complete_energy.append(complete)
            optimal_order.append(ordered)
            ordered = False
            energy_sums.append(energy_sum)
            cluster_ids.append(cluster_count)
            if len(cluster) > 1:
                acceptable.append(
                    pred_cluster[0] == cluster[0] and pred_cluster[1] == cluster[1]
                )
                first_int_good.append(pred_cluster[0] == cluster[0])
            else:
                acceptable.append(True)
            true_cluster_ids.append(cluster_id)
        lengths.extend(l * [l])
        cluster_count += 1

    return (
        features,
        optimal_order,
        complete_energy,
        energy_sums,
        lengths,
        correct_solution_index,
        other_solution_index,
        cluster_ids,
        acceptable,
        true_cluster_ids,
    )


def generate_semi_greedy_data_single_cluster(
    event: Event, cluster: List, width: int = 5, **kwargs
) -> Tuple[List]:
    """
    Feature generation for single events. Get the features for the true order,
    then all other orders, and concatenate each order with the true order.

    Only creates data pairs between the true order and other orders.
    Does partial orders that would be used by a semi-greedy method.
    """
    if isinstance(cluster, dict):
        cluster = list(cluster.values())[0]
    features = []
    optimal_order = []
    acceptable = []
    first_int_good = []
    energy_sum = sum(event.points[i].e for i in cluster)

    num_perms = 0

    if width is None:
        cluster_width = None
    else:
        cluster_width = min(len(cluster), width)
        if len(cluster) == width + 1:  # No savings for eliminating final element
            cluster_width = len(cluster)
    ordered = True
    for pred_cluster in permutations(cluster, r=cluster_width):
        num_perms += 1
        new_features = get_fom_pieces(
            event, pred_cluster, start_energy=energy_sum, **kwargs
        )
        features.append(new_features)
        optimal_order.append(ordered)
        ordered = False
        if len(cluster) > 1:
            acceptable.append(
                pred_cluster[0] == cluster[0] and pred_cluster[1] == cluster[1]
            )
            first_int_good.append(pred_cluster[0] == cluster[0])
        else:
            acceptable.append(True)

    # optimal_solution_index = [0]*num_perms
    # other_solution_index = list(range(num_perms))

    return (
        features,
        optimal_order,
        acceptable,
        first_int_good,
    )
    # optimal_solution_index, other_solution_index)


def build_pairs(ray_ids: np.array, ordered: np.array):
    """
    - ray_ids: query/g-ray labels for data
    - ordered: boolean label if data represents a correctly ordered permutation

    Returns
    - opt_index: index of correctly ordered permutation in the data
    - other_index: indices of all data

    Notes
    - Allows opt_index == other_index
    """
    indices = np.arange(ray_ids.shape[0], dtype=int)
    opt_index = np.zeros(ray_ids.shape, dtype=int)
    for q in np.unique(ray_ids):
        opt_index[indices[ray_ids == q]] = indices[ray_ids == q][
            np.argmax(ordered[indices[ray_ids == q]])
        ]
    return opt_index, indices


def build_data_single_rays(
    single_ray_events: List[Event], single_ray_clusters: List[Dict], width: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a data set using single g-ray events

    - single_ray_events: list of single g-ray events
    - single_ray_clusters: list of single g-ray clustering dicts (only one cluster)
    - width: semi-greedy ordering width (how many interactions ahead to permute)

    Returns
    - ray_features: various FOM/objective function values for the ordered permutation
    - ray_ids: the id of the g-rays
    - ray_ordered: indicator if the data represents the correctly ordered g-ray
    - ray_ordered_acceptable: indicator if the ordering of the g-ray is
    acceptable (first two correct)
    - ray_ordered_first: indicator if the ordering of the g-ray has the first correct
    """
    ray_features = []
    ray_ordered = []
    ray_ordered_acceptable = []
    ray_ordered_first = []
    ray_ids = []
    for (i, event), cluster in zip(enumerate(single_ray_events), single_ray_clusters):
        (
            features,
            ordered,
            acceptable,
            first,
        ) = generate_semi_greedy_data_single_cluster(event, cluster, width=width)
        ray_id = [i] * len(features)
        ray_features.extend(features)
        ray_ordered.extend(ordered)
        ray_ordered_acceptable.extend(acceptable)
        ray_ordered_first.extend(first)
        ray_ids.extend(ray_id)
    ray_ids = np.array(ray_ids)
    ray_features = np.array(ray_features)
    ray_ordered = np.array(ray_ordered)
    ray_ordered_acceptable = np.array(ray_ordered_acceptable)
    ray_ordered_first = np.array(ray_ordered_first)
    return ray_features, ray_ids, ray_ordered, ray_ordered_acceptable, ray_ordered_first


def generate_all_data(
    event: Event,
    true_clusters: Dict[int, List],
    true_energies: Dict[int, float],
    tol: float = 1e-2,
    max_cluster_size: int = 7,
    use_tango: bool = False,
    remove_pair_production: bool = True,
    width: int = None,
    **kwargs,
) -> Tuple[List]:
    """
    Feature generation for single events. Get the features for the true order,
    then all other orders, and concatenate each order with the true order.

    Only creates data pairs between the true order and other orders
    """
    X = []
    y = []
    e = []
    count = 0
    correct_solution_index = []
    index2 = []
    lengths = []
    cid = []
    acceptable = []
    optimal_order = []
    c_count = 1
    for cluster_id, cluster in true_clusters.items():
        if len(cluster) >= max_cluster_size or len(cluster) <= 1:
            continue

        if remove_pair_production:
            # remove pair production interactions and others
            if any(event.points[i].interaction_type > 2 for i in cluster):
                continue

        l = factorial(len(cluster))
        energy_sum = sum(event.points[i].e for i in cluster)
        complete = abs(energy_sum - true_energies[cluster_id]) < tol
        ordered = True
        optimal = True

        # Loop over all possible orders
        index1_start = count
        for pred_cluster in permutations(cluster):
            pred_cluster = list(pred_cluster)
            new_X = get_fom_pieces(event, pred_cluster, **kwargs)
            X.append(new_X)
            lengths.append(l)
            correct_solution_index.append(index1_start)
            index2.append(count)
            count += 1
            y.append([not ordered, not complete, not (ordered and complete)])
            optimal_order.append(optimal)
            optimal = False
            ordered = False
            e.append(energy_sum)
            cid.append(c_count)
            if len(cluster) > 1:
                acceptable.append(
                    pred_cluster[0] == cluster[0] and pred_cluster[1] == cluster[1]
                )
            else:
                acceptable.append(True)
        c_count += 1

    return (
        X,
        y,
        e,
        lengths,
        correct_solution_index,
        index2,
        cid,
        acceptable,
        optimal_order,
    )


def make_data(
    packed_smeared_events: List[Event],
    packed_clusters: List[Dict[int, List]],
    true_energies: Dict = None,
    max_cluster_size: int = 6,
    seed: int = 42,
    test_train_split: float = 0.33,
    debug: bool = False,
    semi_greedy_width: int = None,
    **kwargs,
) -> Tuple[np.ndarray]:
    """
    Method for creating data for multiple events

    Repeatedly invokes `generate_all_data` and manages the indices coming from
    there.

    Returns:
    - features
    - ordered
    - complete
    - energy_sums
    - lengths
    - opt_index
    - other_index
    - cluster_ids
    - acceptable
    - event_ids
    - true_cluster_ids
    """
    if true_energies is None:
        true_energies = m30_true_energies
    features = []
    ordered = []
    complete = []
    energy_sums = []
    lengths = []
    opt_index = []
    other_index = []
    cluster_ids = []
    true_cluster_ids = []
    event_ids = []
    # train_label = []
    acceptable = []
    num_clusters = 0
    count = 0

    # rng = np.random.RandomState(seed)  # RNG for test-train split

    for (event_id, event), cluster in zip(
        enumerate(packed_smeared_events), packed_clusters
    ):
        x, order, comp, e, l, i1, i2, cid, acc, t_cid = generate_semi_greedy_data(
            event,
            cluster,
            true_energies=true_energies,
            max_cluster_size=max_cluster_size,
            width=semi_greedy_width,
            **kwargs,
        )
        if len(x) > 0:
            opt_index.extend([len(features) + i for i in i1])
            other_index.extend([len(features) + i for i in i2])
            features.extend(x)
            ordered.extend(order)
            complete.extend(comp)
            energy_sums.extend(e)
            lengths.extend(l)
            cluster_ids.extend([i + num_clusters for i in cid])
            num_clusters += cid[-1]
            count += 1
            # Randomly assign testing and training data
            # train_label.extend([rng.uniform() < 1 - test_train_split]*len(x))
            acceptable.extend(acc)
            event_ids.extend([event_id] * len(x))
            true_cluster_ids.extend(t_cid)

    features = np.array(features)
    ordered = np.array(ordered)
    complete = np.array(complete)
    energy_sums = np.array(energy_sums)
    lengths = np.array(lengths)
    opt_index = np.array(opt_index)
    other_index = np.array(other_index)
    cluster_ids = np.array(cluster_ids)
    # train_label = np.array(train_label)
    acceptable = np.array(acceptable)
    event_ids = np.array(event_ids)
    true_cluster_ids = np.array(true_cluster_ids)
    if debug:
        print(f"Generated data for {len(np.unique(cluster_ids))} clusters")

    return (
        features,
        ordered,
        complete,
        energy_sums,
        lengths,
        opt_index,
        other_index,
        cluster_ids,
        acceptable,
        event_ids,
        true_cluster_ids,
    )


def clean_residuals(r: np.ndarray, normalize: bool = True):
    """
    Convert residuals to unit vectors and ensure there are no nan's in the data
    """
    r = np.nan_to_num(r)
    if normalize:
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
        r = np.nan_to_num(r)
    r[~np.isfinite(r)] = 0
    return r


def normalize_data(X: np.ndarray, return_std: bool = False):
    """
    Remove NaN values, divide by standard deviation of the features (no need to
    subtract mean, all features will be relative)

    Note that we cannot divide by the standard deviation after creating relative
    features, relative features exist in an unknown distribution
    """
    X[np.isnan(X)] = 0.0
    X[~np.isfinite(X)] = 0.0

    std = np.std(X, axis=0)
    for feature_index in range(X.shape[1]):
        if std[feature_index] > 0:
            X[:, feature_index] = X[:, feature_index] / std[feature_index]
    X[np.isnan(X)] = 0

    if return_std:
        return X, std
    return X


def make_residuals(
    X: np.ndarray[float],
    I: np.ndarray[int],
    opt_index: np.ndarray[int],
    other_index: np.ndarray[int],
    train_label: np.ndarray[bool] = None,
) -> Tuple[np.ndarray]:
    """
    Takes absolute data X and transforms it into residuals for learning the best
    objective function

    @return  (r, I) or (r_train, I_train, r_val, I_val)
    """
    if train_label is None:
        r = np.nan_to_num(X[other_index]) - np.nan_to_num(X[opt_index])
        r = clean_residuals(r)

        mask = other_index != opt_index
        return r[mask], I[mask]

    I_train = I[train_label]
    r_train = np.nan_to_num(X[other_index[train_label]]) - np.nan_to_num(
        X[opt_index[train_label]]
    )
    r_train = clean_residuals(r_train)
    mask_train = other_index[train_label] != opt_index[train_label]

    val_label = np.logical_not(train_label)
    I_val = I[val_label]
    r_val = np.nan_to_num(X[other_index[val_label]]) - np.nan_to_num(
        X[opt_index[val_label]]
    )
    r_val = clean_residuals(r_val)
    mask_val = opt_index[val_label] != other_index[val_label]

    return r_train[mask_train], I_train[mask_train], r_val[mask_val], I_val[mask_val]


def make_train_label(
    I: np.ndarray, val_ratio: float = 0.33, seed: int = 42
) -> np.ndarray:
    """
    Randomly select the training data and validation data
    """
    rng = np.random.RandomState(seed=seed)
    labels = {}
    for i in np.unique(I):
        if rng.uniform() < val_ratio:
            labels[i] = False
        else:
            labels[i] = True
    train_label = np.zeros(I.shape, dtype=bool)
    for i, j in enumerate(I):
        train_label[i] = labels[j]
    return train_label


def cluster_eval(
    events: list[Event],
    clusters: list[dict[int, list[int]]],
    tol: float = 2e-2,
    true_energies: dict = None,
) -> tuple[list, list]:
    """
    Do the clusters represent complete gamma-rays? Do the gamma-rays end with an absorption?

    - events: list of event objects
    - clusters: list of clustering dicts
    - tol: numerical tolerance for matching energy to label a g-ray complete
    - true_energies: true energies for each cluster/g-ray id

    Returns
    - complete_deposits: list of dictionaries indicating if the cluster/g-ray is complete
    - absorptions: list of dictionaries indicating if the cluster/g-ray ends in an absorption
    - pair_productions: list of dictionaries indicating if the cluster/g-ray
    contains a pair production
    """
    use_list = False
    if true_energies is None:
        true_energies = m30_true_energies
    if isinstance(true_energies, list):
        true_energies = np.array(true_energies)
        use_list = True
    complete_deposits = []
    absorptions = []
    pair_productions = []
    for event, clustering in zip(events, clusters):
        energy_sum = event.energy_sums(clustering)
        complete = {}
        absorbed = {}
        pair_prod = {}
        for cluster_id, cluster_energy_sum in energy_sum.items():
            if isinstance(true_energies, dict):
                complete[cluster_id] = (
                    np.abs(true_energies[cluster_id] - cluster_energy_sum) < tol
                )
            elif use_list:
                complete[cluster_id] = (
                    np.min(np.abs(true_energies - cluster_energy_sum)) < tol
                )
            absorbed[cluster_id] = (
                event.points[clustering[cluster_id][-1]].interaction_type == 2
            )
            if event.points[1].interaction_type is not None:
                pair_prod[cluster_id] = any(
                    [
                        event.points[i].interaction_type > 2
                        for i in clustering[cluster_id]
                    ]
                )
            else:
                pair_prod[cluster_id] = False
        complete_deposits.append(complete)
        absorptions.append(absorbed)
        pair_productions.append(pair_prod)
    return complete_deposits, absorptions, pair_productions


def split_g_ray_events(
    events: list[Event],
    clusters: list[dict],
    tol: float = 2e-2,
    true_energies: dict = None,
):
    """
    We want to split the events into individual pieces so we can balance the created data

    - events: g-ray events list
    - clusters: g-ray events clusters list of dicts
    - tol: energy tolerance for determining if the g-ray is a complete deposit
    - true_energies: true energy for each cluster/g-ray id

    Returns
    - ray_energies: list of energy sums from each cluster
    - ray_true_energy: list of true emitted energy from each cluster
    - ray_completeness: list of booleans indicating if the energy deposit is complete
    - ray_absorption: list of booleans indicating if the g-ray ends in absorption
    - ray_pair_production: list of booleans indicating if the g-ray undergoes pair production
    - ray_events: list of events containing just one g-ray
    - ray_clusters: list of clustering dictionaries with just one g-ray
    - ray_cluster_id: list of original cluster ids
    - ray_event_id: list of original event ids
    - ray_length: list of cluster lengths for each g-ray
    """
    if true_energies is None:
        true_energies = m30_true_energies
    completes, absorbs, pair_prods = cluster_eval(
        events, clusters, tol=tol, true_energies=true_energies
    )
    ray_energies = []
    ray_true_energies = []
    ray_completeness = []
    ray_absorption = []
    ray_pair_production = []
    ray_events = []
    ray_clusters = []
    ray_cluster_id = []
    ray_event_id = []
    ray_length = []
    for (event_id, event), cluster, complete, absorb, pair_prod in zip(
        enumerate(events), clusters, completes, absorbs, pair_prods
    ):
        energy_sums = event.energy_sums(cluster)
        ray_energies.extend(energy_sums.values())
        if isinstance(true_energies, dict):
            ray_true_energies.extend([true_energies[k] for k in cluster.keys()])
        else:
            ray_true_energies.extend([0] * len(cluster))
        ray_completeness.extend(complete.values())
        ray_absorption.extend(absorb.values())
        ray_pair_production.extend(pair_prod.values())
        new_events, new_clusters = split_event_clusters(event, cluster)
        ray_events.extend(new_events)
        ray_clusters.extend(new_clusters)
        ray_cluster_id.extend(cluster.keys())
        ray_event_id.extend([event_id] * len(cluster.keys()))
        ray_length.extend([len(a) for a in cluster.values()])
    return (
        np.array(ray_energies),
        np.array(ray_true_energies),
        np.array(ray_completeness),
        np.array(ray_absorption),
        np.array(ray_pair_production),
        np.array(ray_events),
        np.array(ray_clusters),
        np.array(ray_cluster_id),
        np.array(ray_event_id),
        np.array(ray_length),
    )
