"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Data preparation for optimization of the FOM

Generates features for various permutations of the data for use in the
optimization process
"""

from itertools import permutations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import greto.fast_features as ff
from greto.fom_tools import semi_greedy, semi_greedy_batch
from greto.cluster_tools import pack_and_smear_list, cluster_mappings, cluster_linkage
from greto.event_class import Event
from greto.event_tools import split_event_clusters

# from tqdm import tqdm


feature_names = ff.all_feature_names
order_feature_names = ff.order_feature_names
order_feature_boolean_vectors = ff.convert_feature_names_to_boolean_vectors(
    order_feature_names
)

m30_true_energies = {}  # true energies in MeV
for idx in range(1, 31):
    m30_true_energies[idx] = 0.08 + 0.09 * (idx - 1)


# def get_fom_pieces(event: Event, cluster: List, **kwargs):
#     """Get the FOM values from the data"""
#     return list(get_all_features_cluster(event, cluster, **kwargs).values())


def get_ordering_fom_pieces(
    event: Event,
    permutation: Iterable[int],
    event_calc: ff.event_level_values = None,
    start_point: int = 0,
    start_energy: float = None,
    Nmi: int = None,
    bvs: ff.boolean_vectors = None,
    trim_features: bool = False,
    eres: float = 1e-3,
):
    """Get the FOM pieces that can be used for ordering"""
    return ff.get_perm_features(
        event,
        event_calc,
        permutation,
        start_point,
        start_energy,
        Nmi,
        bvs,
        trim_features,
        eres,
    )


def generate_semi_greedy_data(
    event: Event,
    true_clusters: Dict[int, List],
    true_energies: Dict[int, float],
    tol: float = 1e-2,
    max_cluster_size: int = 7,
    remove_pair_production: bool = True,
    width: int = 5,
    **kwargs,  # pylint: disable=unused-argument
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

    event_calc = ff.get_event_level_values(event)
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
            # new_features = get_fom_pieces(
            #     event, pred_cluster, start_energy=energy_sum, **kwargs
            # )
            new_features = get_ordering_fom_pieces(event, pred_cluster, event_calc)
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


def singles_nonsingles_data_creation(
    events: List[Event],
    ordered_clusters_dict: List[Dict],
    debug: bool = True,
    true_energies: Optional[Dict] = None,
    tol: float = 2e-2,
):
    """
    Generate a dictionary of data from ordered clusters for training a
    singles/non-singles classification model.

    Splits the g-ray events into single g-rays. Then generates features for each
    ordered cluster. Clusters should be ordered before being passed to this function.
    """
    if debug:
        print("splitting rays")
    (
        ray_energies,
        ray_true_energy,
        ray_completeness,
        ray_absorption,
        ray_pair_production,
        ray_events,
        ray_clusters,
        ray_cluster_id,
        ray_event_id,
        ray_length,
    ) = split_g_ray_events(
        events, ordered_clusters_dict, tol=tol, true_energies=true_energies
    )

    data = {
        "energy": ray_energies,
        "true_energy": ray_true_energy,
        "completeness": ray_completeness,
        "absorption": ray_absorption,
        "pair_production": ray_pair_production,
        "events": ray_events,
        "clusters": ray_clusters,
        "cluster_ids": ray_cluster_id,
        "event_ids": ray_event_id,
        "length": ray_length,
        "feature_names": ff.all_feature_names,
    }

    if debug:
        print("generating features")
    features = []
    for event, clu in zip(ray_events, ray_clusters):
        event_calc = ff.get_event_level_values(event, None)
        for cluster in clu.values():
            perm_features = ff.get_perm_features(
                event, event_calc, cluster, 0, None, None, None, False
            )
            single_features = ff.get_single_features(
                event, event_calc, cluster, None, False
            )
            cluster_features = ff.get_cluster_features(
                event_calc, cluster, None, None, False
            )
            features.append(
                np.concatenate(
                    (perm_features, single_features, cluster_features), axis=0
                )
            )
    features = np.array(features)

    features[np.isnan(features)] = 0.0
    features[~np.isfinite(features)] = 0.0

    data["features"] = features

    return data


# def generate_single_order_data(
#     event: Event,
#     true_clusters: Dict[int, List],
#     true_energies: Dict[int, float],
#     tol: float = 1e-2,
#     max_cluster_size: int = 7,
#     remove_pair_production: bool = True,
#     width: int = 5,
#     **kwargs,  # pylint: disable=unused-argument
# ) -> Tuple[List]:
#     """
#     Feature generation for single events. Get the features for the true order,
#     then all other orders, and concatenate each order with the true order.

#     Only creates data pairs between the true order and other orders.
#     Does partial orders that would be used by a semi-greedy method.
#     """
#     features = []
#     # y = []
#     complete_energy = []
#     energy_sums = []
#     count = 0
#     correct_solution_index = []
#     other_solution_index = []
#     lengths = []
#     cluster_ids = []
#     true_cluster_ids = []
#     acceptable = []
#     first_int_good = []
#     optimal_order = []
#     cluster_count = 1

#     event_calc = ff.get_event_level_values(event)
#     for cluster_id, cluster in true_clusters.items():
#         if len(cluster) >= max_cluster_size:
#             continue
#         if remove_pair_production:
#             # remove pair production interactions and others; ordering doesn't make sense here
#             if any(event.points[i].interaction_type > 2 for i in cluster):
#                 continue

#         if len(cluster) <= 1:
#             single_features = ff.get_single_features(
#                 event, event_calc, cluster, None, False
#             )
#             perm_features = np.zeros((ff.number_of_feature_values,))
#         else:
#             single_features = np.zeros((ff.number_of_single_feature_values,))
#             perm_features = ff.get_perm_features(
#                 event, event_calc, cluster, 0, None, None, None, False
#             )

#         energy_sum = sum(event.points[i].e for i in cluster)
#         complete = abs(energy_sum - true_energies[cluster_id]) < tol
#         ordered = True

#         features.append(np.concatenate((perm_features, single_features), axis=0))
#         # y.append([not ordered, not complete, not(ordered and complete)])
#         complete_energy.append(complete)
#         optimal_order.append(ordered)
#         energy_sums.append(energy_sum)
#         cluster_ids.append(cluster_count)
#         true_cluster_ids.append(cluster_id)
#         lengths.extend(l * [l])
#         cluster_count += 1

#     return (
#         features,
#         optimal_order,
#         complete_energy,
#         energy_sums,
#         lengths,
#         correct_solution_index,
#         other_solution_index,
#         cluster_ids,
#         acceptable,
#         true_cluster_ids,
#     )


def generate_semi_greedy_data_single_cluster(
    event: Event,
    cluster: List,
    width: int = 5,
    **kwargs,  # pylint: disable=unused-argument
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
    # energy_sum = sum(event.points[i].e for i in cluster)
    event_calc = ff.get_event_level_values(event)

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
        # new_features = get_fom_pieces(
        #     event, pred_cluster, start_energy=energy_sum, **kwargs
        # )
        new_features = get_ordering_fom_pieces(event, pred_cluster, event_calc)
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


# def generate_all_data(
#     event: Event,
#     true_clusters: Dict[int, List],
#     true_energies: Dict[int, float],
#     tol: float = 1e-2,
#     max_cluster_size: int = 7,
#     use_tango: bool = False,
#     remove_pair_production: bool = True,
#     width: int = None,
#     **kwargs,
# ) -> Tuple[List]:
#     """
#     Feature generation for single events. Get the features for the true order,
#     then all other orders, and concatenate each order with the true order.

#     Only creates data pairs between the true order and other orders
#     """
#     X = []
#     y = []
#     e = []
#     count = 0
#     correct_solution_index = []
#     index2 = []
#     lengths = []
#     cid = []
#     acceptable = []
#     optimal_order = []
#     c_count = 1
#     event_calc = ff.get_event_level_values(event)
#     for cluster_id, cluster in true_clusters.items():
#         if len(cluster) >= max_cluster_size or len(cluster) <= 1:
#             continue

#         if remove_pair_production:
#             # remove pair production interactions and others
#             if any(event.points[i].interaction_type > 2 for i in cluster):
#                 continue

#         l = factorial(len(cluster))
#         energy_sum = sum(event.points[i].e for i in cluster)
#         complete = abs(energy_sum - true_energies[cluster_id]) < tol
#         ordered = True
#         optimal = True

#         # Loop over all possible orders
#         index1_start = count
#         for pred_cluster in permutations(cluster):
#             pred_cluster = list(pred_cluster)
#             # new_X = get_fom_pieces(event, pred_cluster, **kwargs)
#             new_X = get_ordering_fom_pieces(event, pred_cluster, event_calc)
#             X.append(new_X)
#             lengths.append(l)
#             correct_solution_index.append(index1_start)
#             index2.append(count)
#             count += 1
#             y.append([not ordered, not complete, not (ordered and complete)])
#             optimal_order.append(optimal)
#             optimal = False
#             ordered = False
#             e.append(energy_sum)
#             cid.append(c_count)
#             if len(cluster) > 1:
#                 acceptable.append(
#                     pred_cluster[0] == cluster[0] and pred_cluster[1] == cluster[1]
#                 )
#             else:
#                 acceptable.append(True)
#         c_count += 1

#     return (
#         X,
#         y,
#         e,
#         lengths,
#         correct_solution_index,
#         index2,
#         cid,
#         acceptable,
#         optimal_order,
#     )


# def make_classification_data(
#     packed_smeared_events: List[Event],
#     packed_clusters: List[Dict[int, List]],
#     true_energies: Dict = None,
#     max_cluster_size: int = 6,
#     # seed: int = 42,
#     # test_train_split: float = 0.33,
#     debug: bool = False,
#     semi_greedy_width: int = None,
#     remove_pair_production: bool = True,
#     **kwargs,
# ) -> Tuple[np.ndarray]:
#     """
#     Method for generating features for prepped events and clusters

#     Repeatedly invokes `generate_all_data` and manages the indices coming from
#     there.

#     Returns:
#     - features
#     - ordered
#     - complete
#     - energy_sums
#     - lengths
#     - opt_index
#     - other_index
#     - cluster_ids
#     - acceptable
#     - event_ids
#     - true_cluster_ids
#     """
#     if true_energies is None:
#         true_energies = m30_true_energies
#     features = []
#     ordered = []
#     complete = []
#     energy_sums = []
#     lengths = []
#     opt_index = []
#     other_index = []
#     cluster_ids = []
#     true_cluster_ids = []
#     event_ids = []
#     # train_label = []
#     acceptable = []
#     num_clusters = 0
#     count = 0

#     # rng = np.random.RandomState(seed)  # RNG for test-train split

#     for (event_id, event), cluster in zip(
#         enumerate(packed_smeared_events), packed_clusters
#     ):
#         x, order, comp, e, l, i1, i2, cid, acc, t_cid = generate_semi_greedy_data(
#             event,
#             cluster,
#             true_energies=true_energies,
#             max_cluster_size=max_cluster_size,
#             width=semi_greedy_width,
#             remove_pair_production=remove_pair_production,
#             **kwargs,
#         )
#         if len(x) > 0:
#             opt_index.extend([len(features) + i for i in i1])
#             other_index.extend([len(features) + i for i in i2])
#             features.extend(x)
#             ordered.extend(order)
#             complete.extend(comp)
#             energy_sums.extend(e)
#             lengths.extend(l)
#             cluster_ids.extend([i + num_clusters for i in cid])
#             num_clusters += cid[-1]
#             count += 1
#             # Randomly assign testing and training data
#             # train_label.extend([rng.uniform() < 1 - test_train_split]*len(x))
#             acceptable.extend(acc)
#             event_ids.extend([event_id] * len(x))
#             true_cluster_ids.extend(t_cid)

#     features = np.array(features)
#     ordered = np.array(ordered)
#     complete = np.array(complete)
#     energy_sums = np.array(energy_sums)
#     lengths = np.array(lengths)
#     opt_index = np.array(opt_index)
#     other_index = np.array(other_index)
#     cluster_ids = np.array(cluster_ids)
#     # train_label = np.array(train_label)
#     acceptable = np.array(acceptable)
#     event_ids = np.array(event_ids)
#     true_cluster_ids = np.array(true_cluster_ids)
#     if debug:
#         print(f"Generated data for {len(np.unique(cluster_ids))} clusters")

#     return (
#         features,
#         ordered,
#         complete,
#         energy_sums,
#         lengths,
#         opt_index,
#         other_index,
#         cluster_ids,
#         acceptable,
#         event_ids,
#         true_cluster_ids,
#     )


def make_data(
    packed_smeared_events: List[Event],
    packed_clusters: List[Dict[int, List]],
    true_energies: Dict = None,
    max_cluster_size: int = 6,
    # seed: int = 42,
    # test_train_split: float = 0.33,
    debug: bool = False,
    semi_greedy_width: int = None,
    remove_pair_production: bool = True,
    **kwargs,
) -> Tuple[np.ndarray]:
    """
    Method for generating features for prepped events and clusters

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
            remove_pair_production=remove_pair_production,
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
        mag_r = np.linalg.norm(r, axis=1, keepdims=True)
        mag_r[mag_r == 0.0] = 1
        r = r / mag_r
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
    rng = np.random.RandomState(seed=seed)  # pylint: disable=no-member
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
                    np.abs(true_energies.get(cluster_id, -10) - cluster_energy_sum)
                    < tol
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
    # include_pair_production: bool = False,
    debug: bool = False,
):
    """
    We want to split the events into individual pieces so we can balance the created data

    - events: g-ray events list
    - clusters: g-ray events clusters list of dicts
    - tol: energy tolerance for determining if the g-ray is a complete deposit
    - true_energies: true energy for each cluster/g-ray id
    # - include_pair_production: include the pair-production g-rays

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
    if debug:
        print(
            f"In split_g_ray_events: num events = {len(events)},"
            + f" num clusters = {len(clusters)}, num completes = {len(completes)}, num absorbs = {len(absorbs)},"
            + f" num pair_prods = {len(pair_prods)}"
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
        # if not include_pair_production:
        #     if pair_prod:
        #         continue
        energy_sums = event.energy_sums(cluster)
        ray_energies.extend(energy_sums.values())
        if isinstance(true_energies, dict):
            ray_true_energies.extend(
                [true_energies.get(k, -10.0) for k in cluster.keys()]
            )
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

    if debug:
        print(
            f"In split_g_ray_events: num events = {len(ray_events)},"
            + f" num completes = {len(completes)},"
            + f" num new completes = {len(ray_completeness)}"
        )

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


def split_g_ray_events_reclustered(
    events: list[Event],
    predicted_clusters: list[dict],
    true_clusters: list[dict],
    tol: float = 2e-2,
    true_energies: dict = None,
    # include_pair_production: bool = False,
):
    """
    We want to split the events into individual pieces so we can balance the created data

    - events: g-ray events list
    - clusters: g-ray events clusters list of dicts
    - tol: energy tolerance for determining if the g-ray is a complete deposit
    - true_energies: true energy for each cluster/g-ray id
    # - include_pair_production: include the pair-production g-rays

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
    true_completes, true_absorbs, _ = cluster_eval(
        events, true_clusters, tol=tol, true_energies=true_energies
    )

    _, _, pair_prods = cluster_eval(
        events, predicted_clusters, tol=tol, true_energies=true_energies
    )

    completes = []
    absorbs = []

    for true_clustering, pred_clustering, true_complete, true_absorb in zip(
        true_clusters, predicted_clusters, true_completes, true_absorbs
    ):
        true_to_pred, pred_to_true = cluster_mappings(true_clustering, pred_clustering)
        complete = {cluster_ID: False for cluster_ID in pred_clustering}
        absorb = {cluster_ID: False for cluster_ID in pred_clustering}
        for cluster_ID in true_clustering:
            if len(true_to_pred[cluster_ID]) == 1:
                if len(pred_to_true[true_to_pred[cluster_ID][0]]) == 1:
                    complete[true_to_pred[cluster_ID][0]] = true_complete[cluster_ID]
                    absorb[true_to_pred[cluster_ID][0]] = true_absorb[cluster_ID]
        completes.append(complete)
        absorbs.append(absorb)

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
        enumerate(events), predicted_clusters, completes, absorbs, pair_prods
    ):
        # if not include_pair_production:
        #     if pair_prod:
        #         continue
        energy_sums = event.energy_sums(cluster)
        ray_energies.extend(energy_sums.values())
        if isinstance(true_energies, dict):
            ray_true_energies.extend(
                [true_energies.get(k, -10.0) for k in cluster.keys()]
            )
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


# %% Column names
fastest_columns = [
    "tango_variance",
    "tango_v_variance",
    "tango_sigma",
    "tango_v_sigma",
    "distances_sum",
    "distances_mean",
    "ge_distances_sum",
    "ge_distances_mean",
    "first_is_not_largest",
    "first_is_not_closest",
]
rsl_columns_1 = [
    "rsl_sum_1",
    "rsl_mean_1",
]
rsl_columns_1v = [
    "rsl_sum_1v",
    "rsl_mean_1v",
    "rsl_wmean_1v",
]
rsl_columns_2 = [
    "rsl_sum_2",
    "rsl_mean_2",
    "rsl_norm_2",
]
rsl_columns_2v = [
    "rsl_sum_2v",
    "rsl_mean_2v",
    "rsl_norm_2v",
    "rsl_wmean_2v",
]

rsl_columns = rsl_columns_1 + rsl_columns_2 + rsl_columns_1v + rsl_columns_2v

fast_columns = (
    fastest_columns
    + rsl_columns
    + [
        "c_penalty_sum_1",
        "c_penalty_mean_1",
        "c_penalty_ell_sum_1",
        "c_penalty_ell_mean_1",
        "c_penalty_ell_sum_2",
        "c_penalty_ell_mean_2",
        "rc_sum_1",
        "rc_mean_1",
        "rc_wmean_1v",
        "rc_norm_2",
        "rc_sum_2",
        "rc_mean_2",
        "rc_wmean_2v",
        "rc_sum_1v",
        "rc_mean_1v",
        "rc_norm_2v",
        "rc_sum_2v",
        "rc_mean_2v",
        "rc_cap_sum_1",
        "rc_cap_mean_1",
        "rc_cap_wmean_1v",
        "rc_cap_norm_2",
        "rc_cap_sum_2",
        "rc_cap_mean_2",
        "rc_cap_wmean_2v",
        "rc_sum_1_penalty_removed",
        "rc_mean_1_penalty_removed",
        "rc_sum_2_penalty_removed",
        "rc_mean_2_penalty_removed",
        "rc_wmean_1v_penalty_removed",
        "rc_wmean_2v_penalty_removed",
        "rc_cap_sum_1v",
        "rc_cap_mean_1v",
        "rc_cap_norm_2v",
        "rc_cap_sum_2v",
        "rc_cap_mean_2v",
        "rc_sum_1v_penalty_removed",
        "rc_mean_1v_penalty_removed",
        "rc_sum_2v_penalty_removed",
        "rc_mean_2v_penalty_removed",
        "rth_sum_1",
        "rth_mean_1",
        "rth_wmean_1v",
        "rth_norm_2",
        "rth_sum_2",
        "rth_mean_2",
        "rth_wmean_2v",
        "rth_sum_1v",
        "rth_mean_1v",
        "rth_norm_2v",
        "rth_sum_2v",
        "rth_mean_2v",
        "rth_cap_sum_1",
        "rth_cap_mean_1",
        "rth_cap_wmean_1v",
        "rth_cap_norm_2",
        "rth_cap_sum_2",
        "rth_cap_mean_2",
        "rth_cap_wmean_2v",
        "rth_sum_1_penalty_removed",
        "rth_mean_1_penalty_removed",
        "rth_sum_2_penalty_removed",
        "rth_mean_2_penalty_removed",
        "rth_wmean_1v_penalty_removed",
        "rth_wmean_2v_penalty_removed",
        "rth_cap_sum_1v",
        "rth_cap_mean_1v",
        "rth_cap_norm_2v",
        "rth_cap_sum_2v",
        "rth_cap_mean_2v",
        "rth_sum_1v_penalty_removed",
        "rth_mean_1v_penalty_removed",
        "rth_sum_2v_penalty_removed",
        "rth_mean_2v_penalty_removed",
    ]
)
fast_tango_columns = [
    "rsl_sum_1v_tango",
    "rsl_mean_1v_tango",
    "rsl_norm_2v_tango",
    "rsl_mean_2v_tango",
    "rsl_sum_2v_tango",
    "rsl_sum_2_tango",
    "rsl_mean_2_tango",
    "rsl_wmean_2v_tango",
    "rsl_mean_1_tango",
    "rsl_sum_1_tango",
    "rsl_norm_2_tango",
    "rsl_wmean_1v_tango",
    "c_penalty_sum_1_tango",
    "c_penalty_mean_1_tango",
    "c_penalty_ell_sum_1_tango",
    "c_penalty_ell_mean_1_tango",
    "c_penalty_ell_sum_2_tango",
    "c_penalty_ell_mean_2_tango",
    "rc_sum_1_tango",
    "rc_mean_1_tango",
    "rc_wmean_1v_tango",
    "rc_norm_2_tango",
    "rc_sum_2_tango",
    "rc_mean_2_tango",
    "rc_wmean_2v_tango",
    "rc_sum_1v_tango",
    "rc_mean_1v_tango",
    "rc_norm_2v_tango",
    "rc_sum_2v_tango",
    "rc_mean_2v_tango",
    "rc_cap_sum_1_tango",
    "rc_cap_mean_1_tango",
    "rc_cap_wmean_1v_tango",
    "rc_cap_norm_2_tango",
    "rc_cap_sum_2_tango",
    "rc_cap_mean_2_tango",
    "rc_cap_wmean_2v_tango",
    "rc_sum_1_penalty_removed_tango",
    "rc_mean_1_penalty_removed_tango",
    "rc_sum_2_penalty_removed_tango",
    "rc_mean_2_penalty_removed_tango",
    "rc_wmean_1v_penalty_removed_tango",
    "rc_wmean_2v_penalty_removed_tango",
    "rc_cap_sum_1v_tango",
    "rc_cap_mean_1v_tango",
    "rc_cap_norm_2v_tango",
    "rc_cap_sum_2v_tango",
    "rc_cap_mean_2v_tango",
    "rc_sum_1v_penalty_removed_tango",
    "rc_mean_1v_penalty_removed_tango",
    "rc_sum_2v_penalty_removed_tango",
    "rc_mean_2v_penalty_removed_tango",
    "rth_sum_1_tango",
    "rth_mean_1_tango",
    "rth_wmean_1v_tango",
    "rth_norm_2_tango",
    "rth_sum_2_tango",
    "rth_mean_2_tango",
    "rth_wmean_2v_tango",
    "rth_sum_1v_tango",
    "rth_mean_1v_tango",
    "rth_norm_2v_tango",
    "rth_sum_2v_tango",
    "rth_mean_2v_tango",
    "rth_cap_sum_1_tango",
    "rth_cap_mean_1_tango",
    "rth_cap_wmean_1v_tango",
    "rth_cap_norm_2_tango",
    "rth_cap_sum_2_tango",
    "rth_cap_mean_2_tango",
    "rth_cap_wmean_2v_tango",
    "rth_sum_1_penalty_removed_tango",
    "rth_mean_1_penalty_removed_tango",
    "rth_sum_2_penalty_removed_tango",
    "rth_mean_2_penalty_removed_tango",
    "rth_wmean_1v_penalty_removed_tango",
    "rth_wmean_2v_penalty_removed_tango",
    "rth_cap_sum_1v_tango",
    "rth_cap_mean_1v_tango",
    "rth_cap_norm_2v_tango",
    "rth_cap_sum_2v_tango",
    "rth_cap_mean_2v_tango",
    "rth_sum_1v_penalty_removed_tango",
    "rth_mean_1v_penalty_removed_tango",
    "rth_sum_2v_penalty_removed_tango",
    "rth_mean_2v_penalty_removed_tango",
]
aft_columns = [
    "c_penalty_sum_1",
    "c_penalty_mean_1",
    "c_penalty_ell_sum_1",
    "c_penalty_ell_mean_1",
    "c_penalty_ell_sum_2",
    "c_penalty_ell_mean_2",
    "rc_sum_1",
    "rc_mean_1",
    "rc_wmean_1v",
    "rc_norm_2",
    "rc_sum_2",
    "rc_mean_2",
    "rc_wmean_2v",
    "rc_sum_1v",
    "rc_mean_1v",
    "rc_norm_2v",
    "rc_sum_2v",
    "rc_mean_2v",
    "rc_cap_sum_1",
    "rc_cap_mean_1",
    "rc_cap_wmean_1v",
    "rc_cap_norm_2",
    "rc_cap_sum_2",
    "rc_cap_mean_2",
    "rc_cap_wmean_2v",
    "rc_sum_1_penalty_removed",
    "rc_mean_1_penalty_removed",
    "rc_sum_2_penalty_removed",
    "rc_mean_2_penalty_removed",
    "rc_wmean_1v_penalty_removed",
    "rc_wmean_2v_penalty_removed",
    "rc_cap_sum_1v",
    "rc_cap_mean_1v",
    "rc_cap_norm_2v",
    "rc_cap_sum_2v",
    "rc_cap_mean_2v",
    "rc_sum_1v_penalty_removed",
    "rc_mean_1v_penalty_removed",
    "rc_sum_2v_penalty_removed",
    "rc_mean_2v_penalty_removed",
    "rth_sum_1",
    "rth_mean_1",
    "rth_wmean_1v",
    "rth_norm_2",
    "rth_sum_2",
    "rth_mean_2",
    "rth_wmean_2v",
    "rth_sum_1v",
    "rth_mean_1v",
    "rth_norm_2v",
    "rth_sum_2v",
    "rth_mean_2v",
    "rth_cap_sum_1",
    "rth_cap_mean_1",
    "rth_cap_wmean_1v",
    "rth_cap_norm_2",
    "rth_cap_sum_2",
    "rth_cap_mean_2",
    "rth_cap_wmean_2v",
    "rth_sum_1_penalty_removed",
    "rth_mean_1_penalty_removed",
    "rth_sum_2_penalty_removed",
    "rth_mean_2_penalty_removed",
    "rth_wmean_1v_penalty_removed",
    "rth_wmean_2v_penalty_removed",
    "rth_cap_sum_1v",
    "rth_cap_mean_1v",
    "rth_cap_norm_2v",
    "rth_cap_sum_2v",
    "rth_cap_mean_2v",
    "rth_sum_1v_penalty_removed",
    "rth_mean_1v_penalty_removed",
    "rth_sum_2v_penalty_removed",
    "rth_mean_2v_penalty_removed",
]
aft_tango_columns = [
    "rc_sum_1_tango",
    "rc_mean_1_tango",
    "rc_wmean_1v_tango",
    "rc_norm_2_tango",
    "rc_sum_2_tango",
    "rc_mean_2_tango",
    "rc_wmean_2v_tango",
    "rc_sum_1v_tango",
    "rc_mean_1v_tango",
    "rc_norm_2v_tango",
    "rc_sum_2v_tango",
    "rc_mean_2v_tango",
    "rc_cap_sum_1_tango",
    "rc_cap_mean_1_tango",
    "rc_cap_wmean_1v_tango",
    "rc_cap_norm_2_tango",
    "rc_cap_sum_2_tango",
    "rc_cap_mean_2_tango",
    "rc_cap_wmean_2v_tango",
    "rc_sum_1_penalty_removed_tango",
    "rc_mean_1_penalty_removed_tango",
    "rc_sum_2_penalty_removed_tango",
    "rc_mean_2_penalty_removed_tango",
    "rc_wmean_1v_penalty_removed_tango",
    "rc_wmean_2v_penalty_removed_tango",
    "rc_cap_sum_1v_tango",
    "rc_cap_mean_1v_tango",
    "rc_cap_norm_2v_tango",
    "rc_cap_sum_2v_tango",
    "rc_cap_mean_2v_tango",
    "rc_sum_1v_penalty_removed_tango",
    "rc_mean_1v_penalty_removed_tango",
    "rc_sum_2v_penalty_removed_tango",
    "rc_mean_2v_penalty_removed_tango",
    "rth_sum_1_tango",
    "rth_mean_1_tango",
    "rth_wmean_1v_tango",
    "rth_norm_2_tango",
    "rth_sum_2_tango",
    "rth_mean_2_tango",
    "rth_wmean_2v_tango",
    "rth_sum_1v_tango",
    "rth_mean_1v_tango",
    "rth_norm_2v_tango",
    "rth_sum_2v_tango",
    "rth_mean_2v_tango",
    "rth_cap_sum_1_tango",
    "rth_cap_mean_1_tango",
    "rth_cap_wmean_1v_tango",
    "rth_cap_norm_2_tango",
    "rth_cap_sum_2_tango",
    "rth_cap_mean_2_tango",
    "rth_cap_wmean_2v_tango",
    "rth_sum_1_penalty_removed_tango",
    "rth_mean_1_penalty_removed_tango",
    "rth_sum_2_penalty_removed_tango",
    "rth_mean_2_penalty_removed_tango",
    "rth_wmean_1v_penalty_removed_tango",
    "rth_wmean_2v_penalty_removed_tango",
    "rth_cap_sum_1v_tango",
    "rth_cap_mean_1v_tango",
    "rth_cap_norm_2v_tango",
    "rth_cap_sum_2v_tango",
    "rth_cap_mean_2v_tango",
    "rth_sum_1v_penalty_removed_tango",
    "rth_mean_1v_penalty_removed_tango",
    "rth_sum_2v_penalty_removed_tango",
    "rth_mean_2v_penalty_removed_tango",
]
oft_columns = [
    "rsg_sum_1",
    "rsg_sum_1_first",
    "rsg_mean_1",
    "rsg_mean_1_first",
    "rsg_wmean_1v",
    "rsg_wmean_1v_first",
    "rsg_norm_2",
    "rsg_sum_2",
    "rsg_sum_2_first",
    "rsg_mean_2",
    "rsg_mean_2_first",
    "rsg_wmean_2v",
    "rsg_wmean_2v_first",
    "rsg_sum_1v",
    "rsg_sum_1v_first",
    "rsg_mean_1v",
    "rsg_mean_1v_first",
    "rsg_norm_2v",
    "rsg_sum_2v",
    "rsg_sum_2v_first",
    "rsg_mean_2v",
    "rsg_mean_2v_first",
    "cross_abs_sum",
    "cross_abs_final",
    "cross_abs_mean",
    "cross_abs_max",
    "cross_abs_ge_dist_sum",
    "cross_abs_ge_dist_final",
    "cross_abs_ge_dist_mean",
    "cross_abs_ge_dist_max",
    "cross_abs_dist_sum",
    "cross_abs_dist_final",
    "cross_abs_dist_mean",
    "cross_abs_dist_max",
    "cross_abs_min",
    "cross_abs_ge_dist_min",
    "cross_abs_dist_min",
    "p_abs_sum",
    "p_abs_final",
    "p_abs_mean",
    "p_abs_max",
    "-log_p_abs_sum",
    "-log_p_abs_final",
    "-log_p_abs_mean",
    "-log_p_abs_max",
    "p_abs_min",
    "-log_p_abs_min",
    "cross_compt_sum",
    "cross_compt_mean",
    "cross_compt_max",
    "cross_compt_ge_dist_sum",
    "cross_compt_ge_dist_mean",
    "cross_compt_ge_dist_max",
    "cross_compt_dist_sum",
    "cross_compt_dist_mean",
    "cross_compt_dist_max",
    "cross_compt_min",
    "cross_compt_ge_dist_min",
    "cross_compt_dist_min",
    "cross_compt_sum_nonfinal",
    "cross_compt_mean_nonfinal",
    "cross_compt_min_nonfinal",
    "cross_compt_dist_sum_nonfinal",
    "cross_compt_dist_mean_nonfinal",
    "cross_compt_dist_min_nonfinal",
    "cross_compt_ge_dist_sum_nonfinal",
    "cross_compt_ge_dist_mean_nonfinal",
    "cross_compt_ge_dist_min_nonfinal",
    "p_compt_sum",
    "p_compt_mean",
    "p_compt_max",
    "p_compt_sum_nonfinal",
    "p_compt_mean_nonfinal",
    "-log_p_compt_sum",
    "-log_p_compt_mean",
    "-log_p_compt_max",
    "-log_p_compt_sum_nonfinal",
    "-log_p_compt_mean_nonfinal",
    "p_compt_min",
    "p_compt_min_nonfinal",
    "-log_p_compt_min",
    "-log_p_compt_min_nonfinal",
    "cross_total_sum",
    "cross_total_mean",
    "cross_total_max",
    "cross_total_ge_dist_sum",
    "cross_total_ge_dist_mean",
    "cross_total_ge_dist_max",
    "cross_total_dist_sum",
    "cross_total_dist_mean",
    "cross_total_dist_max",
    "cross_total_min",
    "cross_total_ge_dist_min",
    "cross_total_dist_min",
]
oft_tango_columns = [
    "rsg_sum_1_tango",
    "rsg_sum_1_first_tango",
    "rsg_mean_1_tango",
    "rsg_mean_1_first_tango",
    "rsg_wmean_1v_tango",
    "rsg_wmean_1v_first_tango",
    "rsg_norm_2_tango",
    "rsg_sum_2_tango",
    "rsg_sum_2_first_tango",
    "rsg_mean_2_tango",
    "rsg_mean_2_first_tango",
    "rsg_wmean_2v_tango",
    "rsg_wmean_2v_first_tango",
    "rsg_sum_1v_tango",
    "rsg_sum_1v_first_tango",
    "rsg_mean_1v_tango",
    "rsg_mean_1v_first_tango",
    "rsg_norm_2v_tango",
    "rsg_sum_2v_tango",
    "rsg_sum_2v_first_tango",
    "rsg_mean_2v_tango",
    "rsg_mean_2v_first_tango",
    "cross_abs_sum_tango",
    "cross_abs_final_tango",
    "cross_abs_mean_tango",
    "cross_abs_max_tango",
    "cross_abs_ge_dist_sum_tango",
    "cross_abs_ge_dist_final_tango",
    "cross_abs_ge_dist_mean_tango",
    "cross_abs_ge_dist_max_tango",
    "cross_abs_dist_sum_tango",
    "cross_abs_dist_final_tango",
    "cross_abs_dist_mean_tango",
    "cross_abs_dist_max_tango",
    "cross_abs_min_tango",
    "cross_abs_ge_dist_min_tango",
    "cross_abs_dist_min_tango",
    "p_abs_sum_tango",
    "p_abs_final_tango",
    "p_abs_mean_tango",
    "p_abs_max_tango",
    "-log_p_abs_sum_tango",
    "-log_p_abs_final_tango",
    "-log_p_abs_mean_tango",
    "-log_p_abs_max_tango",
    "p_abs_min_tango",
    "-log_p_abs_min_tango",
    "cross_compt_sum_tango",
    "cross_compt_mean_tango",
    "cross_compt_max_tango",
    "cross_compt_ge_dist_sum_tango",
    "cross_compt_ge_dist_mean_tango",
    "cross_compt_ge_dist_max_tango",
    "cross_compt_dist_sum_tango",
    "cross_compt_dist_mean_tango",
    "cross_compt_dist_max_tango",
    "cross_compt_min_tango",
    "cross_compt_ge_dist_min_tango",
    "cross_compt_dist_min_tango",
    "cross_compt_sum_nonfinal_tango",
    "cross_compt_mean_nonfinal_tango",
    "cross_compt_min_nonfinal_tango",
    "cross_compt_dist_sum_nonfinal_tango",
    "cross_compt_dist_mean_nonfinal_tango",
    "cross_compt_dist_min_nonfinal_tango",
    "cross_compt_ge_dist_sum_nonfinal_tango",
    "cross_compt_ge_dist_mean_nonfinal_tango",
    "cross_compt_ge_dist_min_nonfinal_tango",
    "p_compt_sum_tango",
    "p_compt_mean_tango",
    "p_compt_max_tango",
    "p_compt_sum_nonfinal_tango",
    "p_compt_mean_nonfinal_tango",
    "-log_p_compt_sum_tango",
    "-log_p_compt_mean_tango",
    "-log_p_compt_max_tango",
    "-log_p_compt_sum_nonfinal_tango",
    "-log_p_compt_mean_nonfinal_tango",
    "p_compt_min_tango",
    "p_compt_min_nonfinal_tango",
    "-log_p_compt_min_tango",
    "-log_p_compt_min_nonfinal_tango",
    "cross_total_sum_tango",
    "cross_total_mean_tango",
    "cross_total_max_tango",
    "cross_total_ge_dist_sum_tango",
    "cross_total_ge_dist_mean_tango",
    "cross_total_ge_dist_max_tango",
    "cross_total_dist_sum_tango",
    "cross_total_dist_mean_tango",
    "cross_total_dist_max_tango",
    "cross_total_min_tango",
    "cross_total_ge_dist_min_tango",
    "cross_total_dist_min_tango",
]
# %% Column name aggregation

column_sets = {
    "fastest": fastest_columns,
    "fast": fast_columns,
    "fast_tango": list(set(fast_columns + fast_tango_columns)),
    "aft": aft_columns,
    "aft_tango": list(set(aft_columns + aft_tango_columns)),
    "aft_fastest": list(set(aft_columns + fastest_columns)),
    "aft_fast": list(set(aft_columns + fast_columns)),
    "aft_fastest_tango": list(set(aft_columns + aft_tango_columns + fastest_columns)),
    "aft_fast_tango": list(
        set(aft_columns + aft_tango_columns + fast_columns + fast_tango_columns)
    ),
    "oft": oft_columns,
    "oft_tango": list(set(oft_columns + oft_tango_columns)),
    "oft_fastest": list(set(oft_columns + fastest_columns)),
    "oft_fast": list(set(oft_columns + fast_columns)),
    "oft_fastest_tango": list(set(oft_columns + oft_tango_columns + fastest_columns)),
    "oft_fast_tango": list(
        set(oft_columns + oft_tango_columns + fast_columns + fast_tango_columns)
    ),
    "aft_true": ["rth_cap_sum_2", "c_penalty_ell_sum_1"],
    "oft_true": [
        "rsg_sum_2v",
        "rsg_sum_2v_first",
        "cross_compt_ge_dist_sum_nonfinal",
        "-log_p_compt_sum_nonfinal",
        "cross_abs_ge_dist_final",
        "-log_p_abs_final",
    ],
}

# %% Predefined methods

methods = {
    "aft": {
        "w": [1.0, 0.4],
        "columns": ["rth_cap_sum_2", "c_penalty_ell_sum_1"],
    },
    "oft": {
        "w": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "columns": [
            "rsg_sum_2v",
            "rsg_sum_2v_first",
            "cross_compt_ge_dist_sum_nonfinal",
            "-log_p_compt_sum_nonfinal",
            "cross_abs_ge_dist_final",
            "-log_p_abs_final",
        ],
    },
}

column_sets["all"] = ff.order_feature_names


# %%


def create_data(
    list_of_events: List[Event],
    list_of_clusters: List[Dict],
    tol: float = 0.02,
    true_energies: dict = None,
    max_clusters_size: int = 6,
    semi_greedy_width: int = 5,
    seed: int = 42,
    packing_distance: float = 0.6,
    energy_threshold: float = 0.005,
    remove_pair_production: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create feature data from pristine events and clusters

    Args:
        - list_of_events: list of gamma-ray events
        - list_of_clusters: list of interaction cluster dictionaries
        - tol: energy tolerance for determining if the g-ray is a full energy
          deposit
        - true_energies: dictionary of the true energies corresponding to each
          cluster_id
        - max_clusters_size: maximum cluster size; larger clusters not processed
        - semi_greedy_width: width of permutation used to generate data
          (complete enumeration for all shorter clusters)
        - seed: random number generator seed
        - packing_distance: interactions closer than this distance [cm] are combined
        - energy_threshold: interactions with energies below this value are deleted

    Returns:
        - df_X: pandas dataframe with features
        - df_Y: pandas dataframe with descriptors of data
    """
    if true_energies is None:
        true_energies = m30_true_energies

    # Split events into individual g-rays
    (
        _,  # energies
        _,  # true_energies,
        _,  # completeness
        _,  # absorption
        _,  # pair_production
        list_of_events,
        list_of_clusters,
        _,  # cluster_id
        _,  # event_id
        _,  # length
    ) = split_g_ray_events(
        list_of_events, list_of_clusters, true_energies=true_energies, tol=tol
    )

    # pack and smear each individual g-ray
    list_of_events, list_of_clusters = pack_and_smear_list(
        list_of_events,
        list_of_clusters,
        packing_distance=packing_distance,
        energy_threshold=energy_threshold,
        seed=seed,
    )

    # reevaluate the individual g-rays
    (
        _,  # energies
        _,  # true_energies,
        _,  # completeness
        _,  # absorption
        _,  # pair_production
        list_of_events,
        list_of_clusters,
        _,  # cluster_id
        _,  # event_id
        _,  # length
    ) = split_g_ray_events(
        list_of_events, list_of_clusters, true_energies=true_energies, tol=tol
    )

    (
        features,
        ordered,
        complete,
        energy_sums,
        lengths,  # not cluster length, length of data related to cluster
        opt_index,
        other_index,
        cluster_ids,
        acceptable,
        event_ids,
        true_cluster_ids,
    ) = make_data(
        list_of_events,
        list_of_clusters,
        true_energies,
        max_clusters_size,
        semi_greedy_width=semi_greedy_width,
        remove_pair_production=remove_pair_production,
    )

    df_X = pd.DataFrame(data=features, columns=ff.order_feature_names)
    df_Y = pd.DataFrame(
        data=np.vstack(
            (
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
        ).transpose(),
        columns=[
            "ordered",
            "complete",
            "energy_sums",
            "lengths",
            "opt_index",
            "other_index",
            "cluster_ids",
            "acceptable",
            "event_ids",
            "true_cluster_ids",
        ],
    )
    return df_X, df_Y


def create_classification_data(
    list_of_events: List[Event],
    list_of_clusters: List[Dict],
    tol: float = 0.02,
    true_energies: dict = None,
    max_clusters_size: int = 6,
    seed: int = 42,
    packing_distance: float = 0.6,
    energy_threshold: float = 0.005,
    use_true: bool = False,
    **order_FOM_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create feature data from pristine events and clusters

    Args:
        - list_of_events: list of gamma-ray events
        - list_of_clusters: list of interaction cluster dictionaries
        - tol: energy tolerance for determining if the g-ray is a full energy
          deposit
        - true_energies: dictionary of the true energies corresponding to each
          cluster_id
        - max_clusters_size: maximum cluster size; larger clusters not processed
        - semi_greedy_width: width of permutation used to generate data
          (complete enumeration for all shorter clusters)
        - seed: random number generator seed
        - packing_distance: interactions closer than this distance [cm] are combined
        - energy_threshold: interactions with energies below this value are deleted

    Returns:
        - df_X: pandas dataframe with features
        - df_Y: pandas dataframe with descriptors of data
    """
    if true_energies is None:
        true_energies = m30_true_energies

    # Split events into individual g-rays
    print("Splitting gamma-rays")
    (
        _,  # energies
        _,  # true_energies,
        completeness,  # completeness
        _,  # absorption
        _,  # pair_production
        list_of_events,
        list_of_clusters,
        _,  # cluster_id
        _,  # event_id
        _,  # length
    ) = split_g_ray_events(
        list_of_events,
        list_of_clusters,
        true_energies=true_energies,
        tol=tol,
    )

    # pack and smear each individual g-ray
    print("Packing and smearing gamma-rays")
    list_of_events, list_of_clusters = pack_and_smear_list(
        list_of_events,
        list_of_clusters,
        packing_distance=packing_distance,
        energy_threshold=energy_threshold,
        seed=seed,
    )

    # reevaluate the individual g-rays
    print("Re-splitting gamma-rays")
    (
        _,  # energies
        _,  # true_energies,
        _,  # completeness
        _,  # absorption
        _,  # pair_production
        list_of_events,
        list_of_clusters,
        _,  # cluster_id
        _,  # event_id
        _,  # length
    ) = split_g_ray_events(
        list_of_events, list_of_clusters, true_energies=true_energies, tol=tol
    )

    # Order the clusters
    ordered_list_of_clusters = []
    first_one = []
    first_two = []
    ordered = []

    print("Ordering clusters")
    for clu, ev in zip(tqdm(list_of_clusters), list_of_events):
        ordered_clusters = {}
        for cluster_id, cluster in clu.items():
            if len(cluster) > 1 and not use_true:
                if order_FOM_kwargs.get("model", None) is not None:
                    ordered_cluster = semi_greedy_batch(
                        ev,
                        cluster,
                        max_cluster_size=max_clusters_size,
                        **order_FOM_kwargs,
                    )
                else:
                    ordered_cluster = semi_greedy(
                        ev,
                        cluster,
                        max_cluster_size=max_clusters_size,
                        **order_FOM_kwargs,
                    )

                ordered_clusters[cluster_id] = ordered_cluster
                if cluster[0] == ordered_cluster[0]:
                    first_one.append(True)
                    if cluster[1] == ordered_cluster[1]:
                        first_two.append(True)
                    else:
                        first_two.append(False)
                else:
                    first_one.append(False)
                    first_two.append(False)
                ordered.append(
                    all(
                        ind == ordered_ind
                        for ind, ordered_ind in zip(cluster, ordered_cluster)
                    )
                )
            else:
                ordered_clusters[cluster_id] = cluster
                first_one.append(True)
                first_two.append(True)
                ordered.append(True)
        ordered_list_of_clusters.append(ordered_clusters)

    print("Creating features")
    data_dict = singles_nonsingles_data_creation(
        list_of_events, ordered_list_of_clusters, true_energies=true_energies
    )

    print("Building dataframes")
    df_X = pd.DataFrame(data=data_dict["features"], columns=data_dict["feature_names"])
    df_Y = pd.DataFrame(
        data=np.vstack(
            (
                data_dict["energy"],
                data_dict["true_energy"],
                data_dict["completeness"],
                data_dict["absorption"],
                data_dict["pair_production"],
                data_dict["cluster_ids"],
                data_dict["event_ids"],
                data_dict["length"],
                first_one,
                first_two,
                ordered,
            )
        ).transpose(),
        columns=[
            "energy_sums",
            "true_energy",
            "complete",
            "absorption",
            "pair_production",
            "cluster_ids",
            "event_ids",
            "lengths",
            "first_one",
            "first_two",
            "ordered",
        ],
    )
    return df_X, df_Y


def create_classification_data_with_clustering(
    list_of_events: List[Event],
    list_of_clusters: List[Dict],
    tol: float = 0.02,
    true_energies: dict = None,
    max_clusters_size: int = 6,
    seed: int = 42,
    packing_distance: float = 0.6,
    energy_threshold: float = 0.005,
    use_true: bool = False,
    alpha_degrees: float = 13.0,
    **order_FOM_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create feature data from pristine events and clusters

    Args:
        - list_of_events: list of gamma-ray events
        - list_of_clusters: list of interaction cluster dictionaries
        - tol: energy tolerance for determining if the g-ray is a full energy
          deposit
        - true_energies: dictionary of the true energies corresponding to each
          cluster_id
        - max_clusters_size: maximum cluster size; larger clusters not processed
        - semi_greedy_width: width of permutation used to generate data
          (complete enumeration for all shorter clusters)
        - seed: random number generator seed
        - packing_distance: interactions closer than this distance [cm] are combined
        - energy_threshold: interactions with energies below this value are deleted

    Returns:
        - df_X: pandas dataframe with features
        - df_Y: pandas dataframe with descriptors of data
    """
    if true_energies is None:
        true_energies = m30_true_energies

    print(f"Reclustering events using alpha_degrees = {alpha_degrees}")
    predicted_clusters = []
    for event in list_of_events:
        predicted_clusters.append(cluster_linkage(event, alpha_degrees=alpha_degrees))

    # Split events into individual g-rays
    print("Splitting gamma-rays")
    (
        _,  # energies
        _,  # true_energies,
        completeness,  # completeness
        _,  # absorption
        _,  # pair_production
        list_of_events,
        list_of_clusters,
        _,  # cluster_id
        _,  # event_id
        _,  # length
    ) = split_g_ray_events_reclustered(
        list_of_events,
        predicted_clusters,
        list_of_clusters,
        true_energies=true_energies,
        tol=tol,
    )

    # print(f"{len(list_of_events)} g-rays and {len(completeness)} complete values")

    # pack and smear each individual g-ray
    print("Packing and smearing gamma-rays")
    list_of_events, list_of_clusters = pack_and_smear_list(
        list_of_events,
        list_of_clusters,
        packing_distance=packing_distance,
        energy_threshold=energy_threshold,
        seed=seed,
        keep_empties=True,
    )

    # Make sure we don't have empty events/clusters
    new_list_of_events = []
    new_list_of_clusters = []
    new_completeness = []
    for event, cluster, complete in zip(list_of_events, list_of_clusters, completeness):
        if len(event.points) > 1:
            new_list_of_events.append(event)
            new_list_of_clusters.append(cluster)
            new_completeness.append(complete)
    list_of_events = new_list_of_events
    list_of_clusters = new_list_of_clusters
    completeness = new_completeness

    print(f"{len(list_of_events)} g-rays and {len(completeness)} complete values")

    # reevaluate the individual g-rays
    print("Re-splitting gamma-rays")
    (
        _,  # energies
        _,  # true_energies,
        _,  # completeness
        _,  # absorption
        _,  # pair_production
        list_of_events,
        list_of_clusters,
        _,  # cluster_id
        _,  # event_id
        _,  # length
    ) = split_g_ray_events(
        list_of_events, list_of_clusters, true_energies=true_energies, tol=tol
    )

    print(f"{len(list_of_events)} g-rays and {len(completeness)} complete values")

    # Order the clusters
    ordered_list_of_clusters = []
    first_one = []
    first_two = []
    ordered = []

    print("Ordering clusters")
    for clu, ev in zip(tqdm(list_of_clusters), list_of_events):
        ordered_clusters = {}
        for cluster_id, cluster in clu.items():
            if len(cluster) > 1 and not use_true:
                if order_FOM_kwargs.get("model", None) is not None:
                    ordered_cluster = semi_greedy_batch(
                        ev,
                        cluster,
                        max_cluster_size=max_clusters_size,
                        **order_FOM_kwargs,
                    )
                else:
                    ordered_cluster = semi_greedy(
                        ev,
                        cluster,
                        max_cluster_size=max_clusters_size,
                        **order_FOM_kwargs,
                    )

                ordered_clusters[cluster_id] = ordered_cluster
                if cluster[0] == ordered_cluster[0]:
                    first_one.append(True)
                    if cluster[1] == ordered_cluster[1]:
                        first_two.append(True)
                    else:
                        first_two.append(False)
                else:
                    first_one.append(False)
                    first_two.append(False)
                ordered.append(
                    all(
                        ind == ordered_ind
                        for ind, ordered_ind in zip(cluster, ordered_cluster)
                    )
                )
            else:
                ordered_clusters[cluster_id] = cluster
                first_one.append(True)
                first_two.append(True)
                ordered.append(True)
        ordered_list_of_clusters.append(ordered_clusters)

    print("Creating features")
    data_dict = singles_nonsingles_data_creation(
        list_of_events, ordered_list_of_clusters, true_energies=true_energies
    )

    print("Building dataframes")
    df_X = pd.DataFrame(data=data_dict["features"], columns=data_dict["feature_names"])
    df_Y = pd.DataFrame(
        data=np.vstack(
            (
                data_dict["energy"],
                # data_dict["true_energy"],
                completeness,
                # data_dict["completeness"],
                data_dict["absorption"],
                data_dict["pair_production"],
                data_dict["cluster_ids"],
                data_dict["event_ids"],
                data_dict["length"],
                first_one,
                first_two,
                ordered,
            )
        ).transpose(),
        columns=[
            "energy_sums",
            # "true_energy",
            "complete",
            "absorption",
            "pair_production",
            "cluster_ids",
            "event_ids",
            "lengths",
            "first_one",
            "first_two",
            "ordered",
        ],
    )
    return df_X, df_Y
